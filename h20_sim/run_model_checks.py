"""Persist P4 unit tests and cross-check source traffic against experiment 005."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from bootstrap import DEFAULT_ROOT, HERE, digest, prepare, timestamp
from operator_source import evidence_path, operator_root
from run_performance import prepare_case


def cross_check(path):
    from run_swiglu import load_module, torch
    torch.set_num_threads(1)
    repo = operator_root()
    folders = {"gqa": "Batch-invariant_GQA", "sink": "Sink_attention", "moe": "FP8_MoE_forward"}
    modules = {key: load_module("sim004_cross_" + key, repo / folder / "analysis005.py") for key, folder in folders.items()}
    totals = dict.fromkeys(folders, 0)
    skipped = []
    for row in json.loads(path.read_text())["cases"]:
        if row.get("task") not in folders or row["status"] != "FUNCTIONAL_PASS":
            continue
        item, program, aux = prepare_case(json.loads(path.read_text()), row)
        if program is None:
            continue
        buffers = {b["name"]: b for b in program["buffers"]}
        works, task = item["work"], row["task"]
        if task == "gqa":
            qo, ko = aux["q_offsets"], aux["kv_offsets"]
            lengths = [(qo[i+1] - qo[i], ko[i+1] - ko[i]) for i in range(len(qo)-1)]
            c = program["launches"][0]["constants"]
            value = modules[task].traffic_case(row["name"], lengths, row["metadata"]["mode"], row["metadata"]["window"],
                    d=c["D"], hq=c["HQ"], hk=c["HK"], dtype="bf16" if buffers["q"]["dtype"] == "torch.bfloat16" else "fp32", bn_override=c["BN"])
            expected = value["total_logical_request_bytes"]
        elif task == "sink":
            q, k = buffers["q"]["shape"], buffers["k"]["shape"]
            value = modules[task].logical_counts((q[0], q[1], k[1], q[2], k[2], q[3]), row["metadata"]["mode"], row["metadata"]["window"],
                                                 element_bytes=2 if buffers["q"]["dtype"] == "torch.bfloat16" else 4)
            key = "tiled" if program["launches"][0]["entry"] == "_sink_tiled" else "scalar"
            expected = value["paths"][key]["logical_bytes"]
        else:
            if buffers["x"]["dtype"] != "torch.bfloat16":
                skipped.append({"case": row["name"], "reason": "005 MoE traffic assumes BF16 activation/output"})
                continue
            value = modules[task].traffic(tuple(row["metadata"]["shape"]), aux["expert_counts"], row["metadata"]["bucket_check"])
            expected = value["program_logical_bytes"]
        actual = sum(w["logical_bytes"] for w in works)
        if actual != expected:
            raise AssertionError(f"{row['name']}: P4={actual}, 005={expected}")
        totals[task] += 1
    if torch.cuda.is_initialized():
        raise RuntimeError("Cross-check initialized CUDA")
    return {"status": "PASS", "cases": totals, "skipped": skipped,
            "evidence_report": str(path), "evidence_sha256": digest(path),
            "counting_modules_sha256": {f"{f}/analysis005.py": digest(repo / f / "analysis005.py") for f in folders.values()},
            "scope": "Cross-implementation agreement on valid source request bytes, not GPU traffic/performance validation"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--evidence", type=evidence_path, required=True, help="Full SIM002 four-task report")
    args = ap.parse_args()
    root = prepare(args.root)
    folder = root / "runs" / f"sim004-checks-{timestamp()}"
    folder.mkdir()
    repo = str(operator_root())
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "TMPDIR": str(folder), "PREROUND_OPERATOR_ROOT": repo, "PYTHONPATH": repo + os.pathsep + str(HERE)}
    tests = {}
    for name, command in (
        ("performance", [sys.executable, "-B", HERE / "test_performance.py"]),
        ("checker", [sys.executable, "-B", "-m", "unittest", "common.test_validation"]),
        ("adapter", [sys.executable, "-B", HERE / "test_adapter.py"]),
        ("pipeline", [sys.executable, "-B", HERE / "test_pipeline.py"]),
    ):
        log = folder / f"{name}.log"
        with log.open("x") as stream:
            result = subprocess.run(command, cwd=HERE.parent, env=env, stdout=stream, stderr=subprocess.STDOUT)
        tests[name] = {"exit_code": result.returncode, "log": str(log), "output": log.read_text(), "sha256": digest(log)}
    report = {"experiment": "sim004", "kind": "model_and_adapter_validation", "gpu_execution": False, "h20_tested": False,
              "run_directory": str(folder), "tests": tests,
              "source_sha256": {p.name: digest(p) for p in (HERE / n for n in ("performance_work.py", "performance_model.py", "run_performance.py", "test_performance.py", "run_model_checks.py"))}}
    try:
        report["cross_check"] = cross_check(args.evidence)
    except Exception as error:
        report["cross_check"] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
    report["status"] = "PASS" if not any(t["exit_code"] for t in tests.values()) and report["cross_check"]["status"] == "PASS" else "FAIL"
    output = HERE / f"results-sim004-checks-{timestamp()[:15]}.json"
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(output), "status": report["status"], "cross_check": report["cross_check"]}), flush=True)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
