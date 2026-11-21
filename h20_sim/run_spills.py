"""Audit offline spill evidence. Unsupported SASS paths remain explicitly unranked."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fnmatch
import json
from pathlib import Path
import subprocess

from run_performance import prepare_case
from operator_source import evidence_key, evidence_path, report_directory
from spill_trace import build_trace, digest, engine_hashes

ROOT = Path("/home/csxsi/out/preround-h20-sim-20261110")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", nargs="+", type=evidence_path, required=True)
    ap.add_argument("--case", default="*")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    reports = [(path, json.loads(path.read_text())) for path in args.reports]
    experiment = "sim006" if all(report.get("experiment") == "sim006" for _, report in reports) else "sim005"
    audit_tasks = {"swiglu", "gqa", "sink", "moe"} if experiment == "sim006" else {"gqa", "router"}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    run = args.root/"runs"/f"{experiment}-spills-{stamp}"
    run.mkdir()
    tool = args.root/"local/cuda-13.1/bin/cuobjdump"
    output = {"experiment": experiment, "kind": "dynamic_spill_evidence", "gpu_execution": False, "h20_tested": False,
              "driver_sha256": digest(__file__), "engine_sha256": engine_hashes(),
              "evidence_reports": {evidence_key(p): digest(p) for p in args.reports}, "cases": []}
    for report_path, report in reports:
        for i, row in enumerate(report["cases"]):
            task = row.get("task") or ("router" if row["name"].startswith("router_") else None)
            if task not in audit_tasks or not fnmatch.fnmatchcase(row["name"], args.case):
                continue
            result = {"name": row["name"], "task": task, "source_report": evidence_key(report_path), "case_index": i, "launches": []}
            try:
                item, program, _ = prepare_case(report, row)
                if program is None:
                    result["status"] = "FUNCTIONAL_GATE_BLOCKED"
                else:
                    for launch in program["launches"]:
                        if not launch["resources"].get("stack", 0) and not launch["resources"].get("local", 0):
                            continue
                        evidence = run/f"{len(output['cases']):03d}-{launch['index']:03d}"
                        evidence.mkdir()
                        result["launches"].append(build_trace(program, launch, Path(row["artifacts"]), evidence, tool))
                    result["status"] = "DYNAMIC_SPILLS_VERIFIED" if result["launches"] else "NO_REPORTED_SPILLS"
            except (ValueError, OSError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
                result.update(status="SPILL_MODEL_REJECTED", reason=f"{type(error).__name__}: {error}")
            output["cases"].append(result)
            print(json.dumps({"case": result["name"], "status": result["status"], "reason": result.get("reason")}), flush=True)
    if not output["cases"]:
        raise ValueError("No functional cases selected for spill audit")
    output["summary"] = dict(Counter(r["status"] for r in output["cases"]))
    path = report_directory(experiment) / f"results-{experiment}-spills-{stamp[:15]}.json"
    with path.open("x") as f:
        json.dump(output, f, indent=2); f.write("\n")
    print(json.dumps({"report": str(path), "summary": output["summary"]}), flush=True)
    return int(bool(output["summary"].get("SPILL_MODEL_REJECTED")))


if __name__ == "__main__":
    raise SystemExit(main())
