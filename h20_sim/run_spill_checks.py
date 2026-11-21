"""Persist P5 tests, evidence mutations, batching invariance and source checks."""

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import unittest

from run_model_checks import cross_check
from operator_source import evidence_key, evidence_path, operator_root
from run_performance import prepare_case
from spill_trace import digest, engine_hashes, load_trace

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/csxsi/out/preround-h20-sim-20261110")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spill-reports", nargs="+", type=evidence_path, required=True)
    ap.add_argument("--regression-report", type=evidence_path, required=True)
    ap.add_argument("--functional-reports", nargs="*", type=evidence_path, default=[])
    args = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    run = ROOT/"runs"/f"sim005-checks-{stamp}"
    run.mkdir()
    stream = io.StringIO()
    sys.path.insert(0, str(operator_root()))
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        "test_spills", "test_performance", "test_adapter", "test_pipeline", "common.test_validation"])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    with (run/"unit-tests.log").open("x") as f:
        f.write(stream.getvalue())
    report = {"experiment": "sim005", "kind": "spill_model_validation", "gpu_execution": False, "h20_tested": False,
              "engine_sha256": engine_hashes(), "tests": {"count": result.testsRun, "failures": len(result.failures),
                 "errors": len(result.errors), "log": str(run/"unit-tests.log")},
              "source_sha256": {p.name: digest(p) for p in HERE.glob("*.py") if p.name.startswith(("spill_", "test_", "performance_", "run_spill"))},
              "spill_reports": {evidence_key(p): digest(p) for p in args.spill_reports}, "mutations": [], "invariance": [], "verified_traces": 0,
              "functional_reports": {evidence_key(p): digest(p) for p in args.functional_reports}, "router_packing": []}
    prepared, groups = [], {}
    for path in args.spill_reports:
        receipt = json.loads(path.read_text())
        for row in receipt["cases"]:
            if row["status"] != "DYNAMIC_SPILLS_VERIFIED":
                continue
            source = evidence_path(row["source_report"])
            if digest(source) != receipt["evidence_reports"][row["source_report"]]:
                raise ValueError("Functional evidence changed")
            source_report = json.loads(source.read_text())
            case = source_report["cases"][row["case_index"]]
            _, program, _ = prepare_case(source_report, case)
            for local in row["launches"]:
                launch = program["launches"][local["launch_index"]]
                trace = load_trace(local, program, launch)
                prepared.append((local, program, launch, trace))
                report["verified_traces"] += 1
                inv = case.get("metadata", {}).get("invariance")
                if inv:
                    groups.setdefault(inv["group"], []).append((inv, trace))
    for name, rows in groups.items():
        bases = [trace for inv, trace in rows if inv["role"] == "base"]
        if len(bases) != 1:
            raise ValueError("Missing unique invariance base")
        base = {g["cta"][0]: g["events"] for g in bases[0]["classes"]}
        for inv, trace in rows:
            if inv["role"] == "base":
                continue
            values = {g["cta"][0]: g["events"] for g in trace["classes"]}
            start, end = inv["output_slice"]
            bstart, bend = inv["base_slice"]
            # PC numbers are compiler placement, not memory access semantics.
            def accesses(events):
                return [e[1:] for e in events]
            if end-start != bend-bstart or any(accesses(values[q]) != accesses(base[bstart+q-start]) for q in range(start, end)):
                raise AssertionError("Batch/chunk changed per-query dynamic local traffic")
            report["invariance"].append({"group": name, "role": inv["role"], "queries": end-start, "passed": True})
    if not prepared:
        raise ValueError("No verified local traces to test")
    receipt, program, launch, trace = prepared[0]
    mutations = {
        "changed_launch": lambda p, l, t: l["grid"].__setitem__(0, l["grid"][0]+1),
        "changed_input": lambda p, l, t: p["buffers"][0].__setitem__("input_sha256", "0"*64),
        "stale_engine": lambda p, l, t: t["engine_sha256"].__setitem__("spill_sass.py", "0"*64),
        "missing_cta_class": lambda p, l, t: t["classes"].pop(),
        "changed_byte_summary": lambda p, l, t: t.__setitem__("load_bytes", t["load_bytes"]+4),
        "changed_site_count": lambda p, l, t: t["static_sites"][0].__setitem__("lane_executions", -1),
        "incomplete_status": lambda p, l, t: t.__setitem__("status", "INCOMPLETE"),
    }
    for name, change in mutations.items():
        p, l, t = copy.deepcopy(program), copy.deepcopy(launch), copy.deepcopy(trace)
        change(p, l, t)
        file = run/f"negative-{name}.json"
        with file.open("x") as f:
            json.dump(t, f)
        r = {**receipt, "trace": str(file), "trace_sha256": digest(file)}
        try:
            load_trace(r, p, l)
        except (ValueError, KeyError) as error:
            report["mutations"].append({"case": name, "rejected": True, "reason": str(error)})
        else:
            raise AssertionError(f"Corrupt evidence accepted: {name}")
    try:
        load_trace({**receipt, "trace_sha256": "0"*64}, program, launch)
    except ValueError:
        report["mutations"].append({"case": "trace_hash", "rejected": True})
    else:
        raise AssertionError("Trace digest mutation accepted")
    report["source_work_cross_check"] = cross_check(args.regression_report)
    if args.functional_reports:
        from pipeline import read_snapshot
        from run_swiglu import torch
        from common.validation import compare_bits
        for path in args.functional_reports:
            functional = json.loads(path.read_text())
            for row in functional["cases"]:
                if row.get("task") != "router" or row["status"] != "FUNCTIONAL_PASS":
                    continue
                _, program, _ = prepare_case(functional, row)
                if not any(l["entry"] == "_packed_gemm" for l in program["launches"]):
                    continue
                folder = Path(row["artifacts"])
                shape = next(b["shape"] for b in program["buffers"] if b["name"] == "w")
                weights = read_snapshot(folder/"w.input.bin", torch.empty(shape, dtype=torch.float32))
                high = weights.bfloat16()
                low = ((weights - high.float())*256).half()
                checks = {}
                for name, expected in (("high", high), ("low", low)):
                    actual = read_snapshot(folder/f"step000-{name}.bin", expected)
                    final = read_snapshot(folder/f"step001-{name}.bin", expected)
                    checks["pack_"+name] = compare_bits(actual, expected, name="pack_"+name)
                    checks["preserved_"+name] = compare_bits(final, actual, name="preserved_"+name)
                report["router_packing"].append({"case": row["name"], "passed": True, "checks": checks})
        if torch.cuda.is_initialized():
            raise RuntimeError("Unexpected physical CUDA initialization")
    report["status"] = "PASS" if result.wasSuccessful() else "FAIL"
    path = HERE/f"results-sim005-checks-{stamp[:15]}.json"
    with path.open("x") as f:
        json.dump(report, f, indent=2); f.write("\n")
    print(json.dumps({"report": str(path), "status": report["status"], "tests": report["tests"],
                      "traces": report["verified_traces"], "negative_controls": len(report["mutations"]),
                      "batch_chunk_checks": len(report["invariance"]), "cross_check": report["source_work_cross_check"]}), flush=True)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
