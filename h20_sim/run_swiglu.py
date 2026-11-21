"""Execute the repository's SM90 SwiGLU PTX on CPU and check real outputs."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from bootstrap import DEFAULT_ROOT, HERE, LOCK, build_environment, digest, prepare, timestamp, validated_build

from operator_source import operator_root, source_key

REPO = operator_root()
sys.path.insert(0, str(REPO))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from common.offline_analysis import parse_resource_usage
from common.validation import compare


class UnsupportedKernel(RuntimeError):
    """The narrow P1 replay contract cannot execute this artifact faithfully."""


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    return value


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(json_safe(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def checked_command(command, cwd, env, log_name, timeout=180):
    log = cwd / log_name
    start = time.monotonic()
    with log.open("x") as output:
        completed = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                                   stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    elapsed = time.monotonic() - start
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}); inspect {log}\n"
                           + log.read_text(errors="replace")[-2500:])
    return log, elapsed


def runtime_environment(root):
    source = root / "upstream/FlashGPU-Sim"
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != LOCK["revision"]:
        raise RuntimeError(f"Backend revision mismatch: {actual}")
    libs = list((source / "lib").glob("gcc-*/cuda-*/release/libcudart.so"))
    if len(libs) != 1:
        raise RuntimeError("Build the pinned backend before running")
    library = libs[0].resolve()
    validated_build(root, source, library)
    env = build_environment(root)
    env.update({"GPGPUSIM_ROOT": str(source),
                "GPGPUSIM_CONFIG": str(library.parent.relative_to(source / "lib")),
                "GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN": "1",
                "PTXAS_CUDA_INSTALL_PATH": env["CUDA_INSTALL_PATH"],
                "LD_LIBRARY_PATH": f"{library.parent}:{env['LD_LIBRARY_PATH']}",
                "OMP_NUM_THREADS": "4"})
    return source, library, env


def copy_config(source, destination):
    for path in (source / LOCK["baseline_config"]).iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)


def raw_tensor(tensor):
    return bytes(tensor.contiguous().view(torch.uint8).flatten().tolist())


def read_tensor(path, dtype, shape):
    count = math.prod(shape)
    size = torch.empty((), dtype=dtype).element_size()
    data = path.read_bytes()
    if len(data) != (count + 32) * size:
        raise ValueError(f"Unexpected output size: {path}")
    if data[count * size:] != b"\xa5" * (32 * size):
        raise AssertionError(f"Output guard overwritten: {path}")
    return torch.frombuffer(bytearray(data[:count * size]), dtype=dtype).reshape(shape).clone()


def numerical_check(actual, expected, tolerance, name):
    try:
        return {**compare(actual, expected, atol=tolerance, rtol=tolerance, name=name), "passed": True}
    except AssertionError as error:
        try:
            details = json.loads(str(error))
        except json.JSONDecodeError:
            return {"passed": False, "error": str(error)}
        first = details.get("first_failure")
        if first is not None:
            index = tuple(first)
            details.update(first_actual=actual[index].double().item(),
                           first_expected=expected[index].double().item())
        return {**details, "passed": False}


def cases(suite):
    seeds = (1108,) if suite == "smoke" else (1108, 9073)
    for seed in seeds:
        for dtype in (torch.float32, torch.bfloat16):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            sizes = (31,) if suite == "smoke" else (0, 1, 31, 257, 1025, 32767, 32768, 32769)
            for n in sizes:
                values = [torch.randn(1, n, generator=generator, dtype=torch.float32).to(dtype)
                          for _ in range(3)]
                yield seed, dtype, f"random_{n}", values
            if suite == "smoke":
                continue
            a = torch.tensor([-100, -20, -8, -1.27846454, -1, 0, 1, 8, 20, 80], dtype=dtype)[None]
            for name, values in (
                ("saturation_and_derivative_cancellation", [a, torch.ones_like(a) * 16, torch.ones_like(a) * 0.125]),
                ("amplified_cancellation_diagnostic", [a, torch.ones_like(a) * 1024, torch.ones_like(a) * 64]),
                ("zero_a", [torch.zeros_like(a), torch.ones_like(a), torch.ones_like(a)]),
                ("zero_b", [a, torch.zeros_like(a), torch.ones_like(a)]),
                ("zero_g", [a, torch.ones_like(a), torch.zeros_like(a)]),
            ):
                yield seed, dtype, name, values


def export_kernel(fn, values, folder, env):
    n = values[0].numel()
    block = 1024 if n >= 32768 else 256
    scalar_type = "fp32" if values[0].dtype == torch.float32 else "bf16"
    signature = {name: f"*{scalar_type}" for name in ("A", "B", "G", "DA", "DB")}
    constants = {"N": n, "BLOCK": block}
    signature.update({name: "constexpr" for name in constants})
    compiled = triton.compile(ASTSource(fn, signature, constexprs=constants),
                              target=GPUTarget("cuda", 90, 32), options={"num_warps": 4})
    ptx = compiled.asm["ptx"]
    entry = compiled.metadata.name
    params = re.search(r"\.entry\s+" + re.escape(entry) + r"\s*\((.*?)\)", ptx, re.S)
    if not params or params.group(1).count(".param") != 7:
        raise UnsupportedKernel("Unsupported PTX argument ABI; expected five pointers and two scratch pointers")
    if compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
        raise UnsupportedKernel("P1 harness does not support nonempty Triton runtime scratch")
    target = re.search(r"(?m)^\s*\.target\s+(\w+)", ptx).group(1)
    if target not in ("sm_90", "sm_90a"):
        raise UnsupportedKernel(f"Unexpected target: {target}")
    (folder / "kernel.ptx").write_text(ptx)
    (folder / "kernel.cubin").write_bytes(compiled.asm["cubin"])
    info_log, _ = checked_command(["cuobjdump", "--dump-resource-usage", "kernel.cubin"],
                                  folder, env, "resources.log")
    allocations = parse_resource_usage(info_log.read_text())
    resources = next(row for row in allocations if row["function"] == entry)
    # P1 requires spill-free kernels: metadata alone cannot reproduce inserted spill instructions.
    if resources.get("stack", 0) or resources.get("local", 0):
        raise UnsupportedKernel("P1 cannot validate a kernel with compiled stack/local storage")
    (folder / "kernel.ptxinfo").write_text(
        f"ptxas info    : 0 bytes gmem\n"
        f"ptxas info    : Compiling entry function '{entry}' for '{target}'\n"
        f"ptxas info    : Function properties for {entry}\n"
        f"    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
        f"ptxas info    : Used {resources['reg']} registers, {resources.get('shared', 0)} bytes smem, 0 bytes cmem[0], 0 bytes lmem\n")
    arch = target.removeprefix("sm_")
    checked_command(["fatbinary", "--64", "--create=kernel.fatbin",
                     f"--image3=kind=elf,sm={arch},file=kernel.cubin",
                     f"--image3=kind=ptx,sm={arch},file=kernel.ptx"], folder, env, "fatbinary.log")
    record = {"entry": entry, "target": target, "block": block, "num_warps": 4,
              "shared_bytes": compiled.metadata.shared, "resources": resources,
              "ptx_version": re.search(r"(?m)^\s*\.version\s+([\d.]+)", ptx).group(1),
              "static_instruction_counts": instruction_counts(ptx),
              "ptx_sha256": digest(folder / "kernel.ptx"), "cubin_sha256": digest(folder / "kernel.cubin")}
    write_json(folder / "export.json", record)
    return record


def instruction_counts(ptx):
    """Static PTX mnemonics; these are neither dynamic counts nor ISA conformance claims."""
    counts = {}
    for line in ptx.splitlines():
        line = line.split("//", 1)[0].strip()
        match = re.match(r"(?:@!?%\w+\s+)?([a-z][\w.]*)(?:\s+[^;]*)?;", line)
        if match:
            op = match.group(1)
            counts[op] = counts.get(op, 0) + 1
    return dict(sorted(counts.items()))


def compile_replay(folder, library, env):
    executable = folder / "replay"
    checked_command(["g++", "-std=c++17", "-O2", f"-I{env['CUDA_INSTALL_PATH']}/include",
                     str(HERE / "replay.cpp"), f"-L{library.parent}", f"-Wl,-rpath,{library.parent}",
                     "-lcudart", "-ldl", "-o", executable], folder, env, "compile-replay.log")
    return executable


def run_case(fn, oracle, values, folder, source, executable, library, env, timeout, config_lines=()):
    n = values[0].numel()
    tolerance = 2e-5 if values[0].dtype == torch.float32 else 1e-2
    expected = oracle.reference(*values)
    independent = oracle.autograd_reference(*values)
    oracle_checks = [numerical_check(a, b, tolerance, f"oracle/{i}")
                     for i, (a, b) in enumerate(zip(expected, independent))]
    if not all(row["passed"] for row in oracle_checks):
        raise RuntimeError("Existing mathematical and autograd references disagree")
    if not n:
        actual = tuple(torch.empty_like(x) for x in values[:2])
        return {"status": "FUNCTIONAL_PASS", "empty_no_launch": True,
                "oracle_checks": oracle_checks,
                "checks": [numerical_check(a, b, tolerance, f"empty/{i}")
                           for i, (a, b) in enumerate(zip(actual, expected))]}
    export = export_kernel(fn, values, folder, env)
    copy_config(source, folder)
    if config_lines:
        with (folder / "gpgpusim.config").open("a") as stream:
            stream.write("\n" + "\n".join(config_lines) + "\n")
    element_size = values[0].element_size()
    names = ("a", "b", "g", "da", "db")
    for i, name in enumerate(names):
        contents = raw_tensor(values[i]) if i < 3 else b"\xff" * (n * element_size)
        (folder / f"{name}.bin").write_bytes(contents + b"\xa5" * (32 * element_size))
    # The pinned loader recognizes a .cubin's adjacent .ptx sidecar. Its .fatbin
    # fallback instead scans the host executable, which contains no device code.
    log, wall_time = checked_command(["stdbuf", "-oL", "-eL", executable, folder / "kernel.cubin", export["entry"],
                                      n, element_size, export["block"], export["shared_bytes"], library],
                                     folder, env, "simulation.log", timeout)
    text = log.read_text(errors="replace")
    if "REPLAY_OUTPUTS_WRITTEN" not in text:
        raise RuntimeError("Simulator did not finish output serialization")
    if "this function has not been implemented" in text:
        raise RuntimeError("Replay used an unimplemented backend API")
    for name in names[:3]:
        if (folder / f"{name}.bin").read_bytes() != (folder / f"{name}.out.bin").read_bytes():
            raise AssertionError(f"Input or input guard modified: {name}")
    maps = (folder / "loaded-libraries.txt").read_text()
    if str(library) not in maps or re.search(r"/[^\s]*libcuda\.so", maps):
        raise AssertionError("Unexpected CUDA driver library mapping")
    actual = [read_tensor(folder / f"{name}.out.bin", values[0].dtype, values[0].shape) for name in names[3:]]
    checks = [numerical_check(a, b, tolerance, name) for a, b, name in zip(actual, expected, names[3:])]
    bad = expected[0].clone()
    bad.flatten()[0] = 1e6
    negative_rejected = not numerical_check(bad, expected[0], tolerance, "negative_control")["passed"]
    if not negative_rejected:
        raise AssertionError("Checker accepted deliberately corrupted output")
    matches = re.findall(r"(?m)^gpu_tot_sim_cycle\s*=\s*(\d+)", text)
    return {"status": "FUNCTIONAL_PASS" if all(row["passed"] for row in checks) else "FUNCTIONAL_FAIL",
            "checks": checks, "oracle_checks": oracle_checks, "export": export, "input_unchanged": True, "guards_intact": True,
            "negative_control_rejected": negative_rejected, "physical_driver_loaded": False,
            "simulated_cycles_upstream_h100": int(matches[-1]) if matches else None,
            "simulator_wall_seconds": wall_time, "config_sha256": digest(folder / "gpgpusim.config"),
            "input_sha256": {name: digest(folder / f"{name}.bin") for name in names[:3]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA must not be initialized")
    root = prepare(args.root)
    run = root / "runs" / f"swiglu-{timestamp()}"
    run.mkdir()
    os.environ["TRITON_CACHE_DIR"] = str(run / "triton-cache")
    source, library, env = runtime_environment(root)
    executable = compile_replay(run, library, env)
    solution = load_module("sim001_swiglu_solution", REPO / "SwiGLU_backward/solution.py")
    oracle = load_module("sim001_swiglu_reference", REPO / "SwiGLU_backward/reference.py")
    report = {"experiment": "sim001", "kind": "swiglu_offline_simulation", "suite": args.suite,
              "run_directory": str(run), "backend": LOCK, "library_sha256": digest(library),
              "build": validated_build(root, source, library),
              "torch": torch.__version__, "triton": triton.__version__, "python": sys.version,
              "gpu_execution": False, "h20_tested": False, "timing_status": LOCK["timing_status"],
              "source_sha256": {source_key(p): digest(p) for p in
                                [REPO / "SwiGLU_backward/solution.py", REPO / "SwiGLU_backward/reference.py",
                                 REPO / "common/validation.py", Path(__file__), HERE / "replay.cpp"]},
              "cases": []}
    for i, (seed, dtype, name, values) in enumerate(cases(args.suite)):
        folder = run / f"{i:03d}-{str(dtype).split('.')[-1]}-{name}"
        folder.mkdir()
        row = {"seed": seed, "dtype": str(dtype), "name": name, "shape": list(values[0].shape),
               "artifacts": str(folder)}
        try:
            row.update(run_case(solution._backward, oracle, values, folder, source,
                                executable, library, env, args.timeout))
        except UnsupportedKernel as error:
            row.update(status="UNSUPPORTED", error=str(error))
        except subprocess.TimeoutExpired as error:
            row.update(status="SIMULATION_TIMEOUT", error=str(error))
        except Exception as error:
            row.update(status="EXECUTION_ERROR", error=f"{type(error).__name__}: {error}")
        report["cases"].append(row)
        write_json(folder / "result.json", row)
        print(json.dumps({"case": i, "name": name, "dtype": str(dtype), "status": row["status"],
                          "cycles": row.get("simulated_cycles_upstream_h100"), "error": row.get("error")}), flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("Export unexpectedly initialized CUDA")
    report["cuda_initialized"] = False
    report["summary"] = {status: sum(row["status"] == status for row in report["cases"])
                         for status in sorted({row["status"] for row in report["cases"]})}
    report["status"] = "PASS" if all(row["status"] == "FUNCTIONAL_PASS" for row in report["cases"]) else "HAS_FAILURES"
    output = args.output or HERE / f"results-sim001-{args.suite}-{timestamp()[:15]}.json"
    write_json(output, report)
    print(json.dumps({"report": str(output), "summary": report["summary"]}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
