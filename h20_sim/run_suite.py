"""Execute P2 GQA/Sink/MoE/Router paths on CPU with original accuracy gates."""

import argparse
import fnmatch
import json
import os
import re
from pathlib import Path
import subprocess

from bootstrap import DEFAULT_ROOT, HERE, LOCK, digest, prepare, timestamp, validated_build
from pipeline import compile_harness, execute
from operator_source import source_key
from run_swiglu import numerical_check, runtime_environment, torch, triton, write_json
from workloads import FOLDERS, REPO, Workloads
from common.validation import compare_bits


def bucket_check(program, outputs):
    ids = program.buffers["ids"]["tensor"]
    count, routes = outputs["count"], outputs["routes"]
    expected = torch.bincount(ids.flatten().long(), minlength=count.numel()).to(torch.int32)
    if not torch.equal(count, expected):
        raise AssertionError("Expert bucket counts differ from routes")
    for expert, size in enumerate(count.tolist()):
        actual = routes[expert, :size].sort().values
        target = (ids.flatten() == expert).nonzero().flatten().to(torch.int32)
        if not torch.equal(actual, target):
            raise AssertionError(f"Missing/duplicated/wrong route in expert {expert}")
    return {"passed": True, "routes_checked": ids.numel(), "counts": count.tolist(),
            "order_assumption": "none; compare membership, not atomic arrival order"}


def check_case(case, outputs, invariant_outputs):
    checks = {name: numerical_check(outputs[name], value, case.tolerance, name)
              for name, value in case.expected.items()}
    if case.original_check is not None and case.expected["output"].numel():
        try:
            checks["original_router_checker"] = {"passed": True, **case.original_check(outputs["output"], case.expected["output"])}
        except AssertionError as error:
            checks["original_router_checker"] = {"passed": False, "error": str(error)}
    if case.metadata.get("bucket_check"):
        try:
            checks["buckets"] = bucket_check(case.program, outputs)
        except AssertionError as error:
            checks["buckets"] = {"passed": False, "error": str(error)}
    invariant = case.metadata.get("invariance")
    if invariant:
        a, b = invariant["output_slice"]
        current = outputs["output"][a:b]
        group = invariant["group"]
        if invariant["role"] == "base":
            invariant_outputs[group] = current.clone()
        elif group not in invariant_outputs:
            checks["invariance"] = {"passed": False, "error": "Base case was not executed; include it in the case filter"}
        else:
            a, b = invariant["base_slice"]
            try:
                checks["invariance"] = {"passed": True, **compare_bits(current, invariant_outputs[group][a:b], name=case.name)}
            except AssertionError as error:
                checks["invariance"] = {"passed": False, "error": str(error)}
    negative_checks = []
    for name, value in case.expected.items():
        if value.numel():
            wrong = value.clone()
            wrong.flatten()[0] = float("nan")
            negative_checks.append(not numerical_check(wrong, value, case.tolerance, "negative_control")["passed"])
    if negative_checks and not all(negative_checks):
        raise AssertionError("Corrupted output was accepted")
    return {"status": "FUNCTIONAL_PASS" if all(c["passed"] for c in checks.values()) else "FUNCTIONAL_FAIL",
            "checks": checks, "negative_control_rejected": all(negative_checks) if negative_checks else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--suite", choices=("smoke", "full", "extended"), default="smoke")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--case", default="*", help="fnmatch filter; include the base case for GQA invariance")
    parser.add_argument("--timeout", type=int, default=180, help="seconds per complete multi-kernel process")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("Physical CUDA must not be initialized")
    root = prepare(args.root)
    run = root / "runs" / f"sim002-{args.suite}-{timestamp()}"
    run.mkdir()
    os.environ["TRITON_CACHE_DIR"] = str(run / "triton-cache")
    source, library, env = runtime_environment(root)
    executable = compile_harness(run, library, env)
    seeds = args.seeds or ([1108, 9073] if args.suite == "full" else [1108])
    files = [HERE / n for n in ("run_suite.py", "workloads.py", "pipeline.py", "pipeline.cpp", "patch_backend.py")]
    for folder in FOLDERS.values():
        files.extend(REPO / folder / n for n in ("solution.py", "reference.py") if (REPO / folder / n).exists())
    files.extend([REPO / "Router_GEMM_prefill/benchmark.py", REPO / "common/validation.py"])
    report = {"experiment": "sim002", "kind": "multi_workload_cpu_simulation", "suite": args.suite,
              "case_filter": args.case, "seeds": seeds, "timeout_seconds": args.timeout,
              "backend": LOCK, "build": validated_build(root, source, library), "run_directory": str(run),
              "torch": torch.__version__, "triton": triton.__version__, "gpu_execution": False, "h20_tested": False,
              "source_sha256": {source_key(p): digest(p) for p in files}, "cases": []}
    invariant_outputs = {}
    for seed in seeds:
        for case in Workloads(seed).cases(args.suite):
            if not fnmatch.fnmatchcase(case.name, args.case):
                continue
            folder = run / f"{len(report['cases']):03d}-{seed}-{case.name}"
            folder.mkdir()
            row = {"name": case.name, "seed": seed, "task": case.task, "dispatch": case.program.dispatch,
                   "metadata": case.metadata, "artifacts": str(folder)}
            try:
                evidence, outputs = execute(case.program, folder, source, library, env, executable, args.timeout)
                row.update(evidence)
                row.update(check_case(case, outputs, invariant_outputs))
            except subprocess.TimeoutExpired as error:
                row.update(status="SIMULATION_TIMEOUT", error=str(error))
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                status = "UNSUPPORTED" if any(w in message.lower() for w in
                                              ("unsupported", "not implemented", "not supported", "parse error")) else "EXECUTION_ERROR"
                row.update(status=status, error=message)
            row["simulation_process_started"] = (folder / "simulation.log").exists()
            if "completed_launches" not in row:
                log = (folder / "simulation.log").read_text(errors="replace") if row["simulation_process_started"] else ""
                row["completed_launches"] = len(re.findall(r"(?m)^SIM002_STEP_COMPLETE\s+\d+", log))
            row["exports"] = [json.loads(p.read_text()) for p in sorted(folder.glob("export*.json"))]
            write_json(folder / "result.json", row)
            report["cases"].append(row)
            print(json.dumps({"case": case.name, "seed": seed, "status": row["status"],
                              "launches": row.get("completed_launches", 0), "error": row.get("error", "")[-1300:]}), flush=True)
    if not report["cases"]:
        raise ValueError("Case filter selected no cases")
    if torch.cuda.is_initialized():
        raise RuntimeError("Export unexpectedly initialized CUDA")
    report["cuda_initialized"] = False
    report["summary"] = {s: sum(r["status"] == s for r in report["cases"])
                         for s in sorted({r["status"] for r in report["cases"]})}
    report["status"] = "PASS" if all(r["status"] == "FUNCTIONAL_PASS" for r in report["cases"]) else "HAS_FAILURES"
    output = args.output or HERE / f"results-sim002-{args.suite}-{timestamp()[:15]}.json"
    write_json(output, report)
    print(json.dumps({"report": str(output), "summary": report["summary"]}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
