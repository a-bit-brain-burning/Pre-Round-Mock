"""Replay one pinned public H200 trace on CPU; not a functional or H20 test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

TRACE_NAME = "kernel-73-ctx_0xc37ce90.tracez"
TRACE_SHA = "5056bb58b9c5350460aa597119c11f66b88cb1345d8bd6417306f1606493d827"
COMMITS = {"framework": "1c9cc831c5d948f89fa9dc8f21f6c87952f5f2ea",
           "backend": "c24e8858eee552b3bc2919b147f14e2a88571775"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(log, returncode, syscalls):
    values = {}
    for key in ("gpu_tot_sim_cycle", "gpu_tot_sim_insn", "gpu_tot_issued_cta"):
        found = re.findall(rf"^{key}\s*=\s*(\d+)\s*$", log, re.M)
        values[key] = int(found[-1]) if found else None
    completed = re.findall(r"^ctas_completed\s+(\d+),", log, re.M)
    values["ctas_completed"] = int(completed[-1]) if completed else None
    checks = {
        "process_success": returncode == 0,
        "normal_exit": "GPGPU-Sim: *** exit detected ***" in log,
        "all_64_ctas_issued": values["gpu_tot_issued_cta"] == 64,
        "all_64_ctas_completed": values["ctas_completed"] == 64,
        "positive_cycles": (values["gpu_tot_sim_cycle"] or 0) > 0,
        "positive_instructions": (values["gpu_tot_sim_insn"] or 0) > 0,
        "expected_kernel": bool(re.search(r"^kernel_name = _Z11scal_kernel", log, re.M)),
        "system_calls_recorded": bool(syscalls.strip()),
        "no_gpu_access_observed": not bool(re.search(r"/dev/(?:dxg|nvidia)|libcuda(?:rt)?\.so", syscalls)),
    }
    return {"status": "TRACE_REPLAY_PASS" if all(checks.values()) else "TRACE_REPLAY_FAIL",
            "checks": checks, "counters": values}


def self_test():
    fixture = ("kernel_name = _Z11scal_kernel_fixture\n"
               "gpu_tot_sim_cycle = 5753\ngpu_tot_sim_insn = 1409024\n"
               "gpu_tot_issued_cta = 64\nctas_completed 64, Shader 0\nGPGPU-Sim: *** exit detected ***\n")
    assert summarize(fixture, 0, "openat(trace)")["status"] == "TRACE_REPLAY_PASS"
    rejected = [(fixture, 1, "openat(trace)"), (fixture.replace(" = 64", " = 63"), 0, "openat(trace)"),
                (fixture.replace("*** exit detected ***", "interrupted"), 0, "openat(trace)"),
                (fixture, 0, 'openat("/dev/dxg")'), (fixture, 0, ""), ("", 0, "openat(trace)"),
                (fixture.replace("ctas_completed 64", "ctas_completed 63"), 0, "openat(trace)")]
    assert all(summarize(*case)["status"] == "TRACE_REPLAY_FAIL" for case in rejected)
    print(json.dumps({"self_test": "PASS", "valid_fixture": 1, "rejected_invalid_fixtures": len(rejected)}))


def replay(base, trace_dir, output, timeout):
    if output.exists():
        raise FileExistsError(f"Preserve existing report: {output}")
    front = base / "accel/gpu-simulator"
    binary = base / "build/accel-sim.out"
    trace = trace_dir / TRACE_NAME
    if digest(trace) != TRACE_SHA:
        raise ValueError("Trace hash differs from the pinned public sample")
    for name, folder in (("framework", base / "accel"), ("backend", front / "gpgpu-sim")):
        actual = subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True).strip()
        if actual != COMMITS[name]:
            raise ValueError(f"Unexpected {name} commit: {actual}")
        if subprocess.check_output(["git", "-C", str(folder), "diff", "HEAD", "--"], text=True):
            raise ValueError(f"Tracked {name} source was modified")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    run = base / f"run-{run_id}"
    run.mkdir()
    for source in (front / "gpgpu-sim/configs/tested-cfgs/SM90_H200").iterdir():
        if source.is_file():
            shutil.copy2(source, run / source.name)
    shutil.copy2(front / "configs/tested-cfgs/SM90_H200/trace.config", run / "trace.config")
    # Accel-Sim recognizes lines starting with "kernel", relative to this list.
    # An absolute path is silently ignored upstream, resulting in a zero-kernel exit.
    (run / TRACE_NAME).symlink_to(trace)
    (run / "kernelslist.g").write_text(TRACE_NAME + "\n", encoding="utf-8")
    command = [str(binary), "-config", str(run / "gpgpusim.config"),
               "-config", str(run / "trace.config"), "-trace", str(run / "kernelslist.g")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    env.pop("LD_PRELOAD", None)
    start = time.monotonic()
    timed_out = False
    with (run / "simulation.log").open("x") as stream:
        process = subprocess.Popen(["strace", "-f", "-e", "trace=openat,ioctl", "-o",
                                    str(run / "syscalls.log"), *command],
                                   cwd=run, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            timed_out, code = True, None
    elapsed = time.monotonic() - start
    log = (run / "simulation.log").read_text(errors="replace")
    syscalls = (run / "syscalls.log").read_text(errors="replace")
    report = {**summarize(log, code, syscalls), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "returncode": code, "timeout": timed_out, "host_wall_seconds_including_strace": elapsed,
              "modeled_gpu": "unmodified upstream SM90_H200", "h20_tested": False,
              "numerical_output_validation": "NOT_PERFORMED: SASS trace performance replay",
              "scope": "one public kernel, not the four custom operators or their E2E",
              "source_commits": COMMITS, "command": command, "run_directory": str(run),
              "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
              "binary_sha256": digest(binary), "trace_sha256": digest(trace),
              "runner_sha256": digest(__file__),
              "config_sha256": {p.name: digest(p) for p in run.iterdir() if p.suffix in (".config", ".icnt")},
              "log_sha256": {name: digest(run / name) for name in ("simulation.log", "syscalls.log")},
              "limitations": ["Public H200 trace, no H20 calibration or independent numerical oracle.",
                              "Standalone kernel has a fresh simulated cache; not the full captured workload.",
                              "Range-extracted trace is locally hashed; full release archive SHA was not verified.",
                              "No TMA/WGMMA/FP64 feature coverage inferred from this scalar kernel."]}
    if timed_out:
        report["status"] = "SIMULATION_TIMEOUT"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "status": report["status"], "counters": report["counters"]}))
    return report["status"] == "TRACE_REPLAY_PASS"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, default=Path("/var/tmp/preround-sim006-build"))
    parser.add_argument("--trace-dir", type=Path, default=Path("/var/tmp/preround-sim006-trace"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.self_test:
        self_test()
    else:
        if args.output is None:
            parser.error("--output must be a new report path")
        if not replay(args.build_root.resolve(), args.trace_dir.resolve(), args.output, args.timeout):
            raise SystemExit(1)
