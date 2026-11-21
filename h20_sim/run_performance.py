"""Analyze verified five-task replay reports with an explicit H20 sensitivity model."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fnmatch
import json
import math
from pathlib import Path
import re
import struct

from performance_model import PROFILE, audit_calls, compare, evaluate, load_profile, scenarios, sha
from contest_scoring import PROBE, contest_profile, forecast_card, load_probe, read_json
from performance_work import WIDTH, account
from operator_source import evidence_key, evidence_path, operator_root, report_directory

HERE = Path(__file__).resolve().parent
CONTRACTS = {
    "swiglu": ("SwiGLU_backward", "c23d3bf921a350df4e0d7e95ccdf79d62b6d8c3a1ec87bcc30fbc921d7984995"),
    "gqa": ("Batch-invariant_GQA", "61bef3c4b3c17d0880ffdd2d21b95129747357f85ca7b855ff57cb0a2898a445"),
    "sink": ("Sink_attention", "2efc31f39c7800d957fb8179d6a9119b789c1c6cc91d032ece1ab828b228912a"),
    "moe": ("FP8_MoE_forward", "e0a0bd69414d867fdb5504a30edb0342927babcc0f5afa69a3301accc2838708"),
    "router": ("Router_GEMM_prefill", "47a864ad54e483e5e379ed7ee34121d6426ce273d852b4592d4d3af1c3cd46d6"),
}


def checked_hash(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Evidence hash mismatch: {path}")
    return actual


def source_gate(report, task):
    if task not in CONTRACTS:
        raise ValueError("Only the five preliminary-contest tasks are supported")
    folder, contract = CONTRACTS[task]
    solution = f"{folder}/solution.py"
    oracle = f"{folder}/{'benchmark.py' if task == 'router' else 'reference.py'}"
    source_hashes = report["source_sha256"]
    if source_hashes.get(solution) != contract:
        raise ValueError("Work accounting has not been reviewed for this solution source")
    repo = operator_root()
    checked_hash(repo / solution, contract)
    checked_hash(repo / oracle, source_hashes[oracle])
    if "common/validation.py" in source_hashes:
        checked_hash(repo / "common/validation.py", source_hashes["common/validation.py"])
    if report.get("gpu_execution") is not False or report.get("h20_tested") is not False:
        raise ValueError("Expected an offline functional simulation report")
    return {solution: contract, oracle: source_hashes[oracle]}


def verified_program(row):
    """Bind numeric claims to the saved manifest, compiled code and snapshots."""
    program = row["program"]
    folder = Path(row["artifacts"])
    if program != json.loads((folder / "program.json").read_text()):
        raise ValueError("Report and artifact manifests differ")
    if program["protocol"] != "SIM002_V1" or program["guard_bytes_each_side"] != 128:
        raise ValueError("Unsupported replay protocol")
    blobs = {}
    for b in program["buffers"]:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", b["name"]):
            raise ValueError("Invalid buffer name")
        expected_bytes = math.prod(b["shape"]) * WIDTH[b["dtype"]]
        if b["bytes"] != expected_bytes:
            raise ValueError("Buffer shape, dtype and byte count disagree")
        path = folder / (b["name"] + ".input.bin")
        checked_hash(path, b["input_sha256"])
        data = path.read_bytes()
        if len(data) != b["bytes"] + 256 or data[:128] != b"\xa5" * 128 or data[-128:] != b"\xa5" * 128:
            raise ValueError("Invalid serialized input/guards")
        if b["name"] in ("ids", "qoff", "koff"):
            blobs[b["name"]] = list(struct.unpack("<" + "i" * (b["bytes"] // 4), data[128:-128]))
    call_audits = {}
    for i, launch in enumerate(program["launches"]):
        if launch["module"] != f"kernel{i:03d}.cubin" or launch["target"] not in ("sm_90", "sm_90a"):
            raise ValueError("Unsupported module or target")
        if launch != json.loads((folder / f"export{i:03d}.json").read_text()):
            raise ValueError("Export record differs from manifest")
        checked_hash(folder / launch["module"], launch["cubin_sha256"])
        checked_hash(folder / f"kernel{i:03d}.ptx", launch["ptx_sha256"])
        if launch["resources"].get("stack", 0) or launch["resources"].get("local", 0):
            actual = [line.strip() for line in (folder / f"sass{i:03d}.log").read_text().splitlines()
                      if re.search(r"\b(?:LDL|STL|CALL|RET)\b", line)]
            if actual != launch.get("sass_stack_or_call_instructions"):
                raise ValueError("SASS spill evidence changed")
            call_audits[i] = audit_calls(launch, (folder / f"sass{i:03d}.log").read_text())
    if program["launches"]:
        checks = row["checks"]
        checks = list(checks.values()) if isinstance(checks, dict) else checks
        if not checks or not all(c.get("passed") is True for c in checks):
            raise ValueError("Numerical checks contradict pass status")
        if row.get("negative_control_rejected") is not True:
            raise ValueError("Missing numerical negative control")
        if row.get("physical_driver_loaded") is not False or row.get("guards_intact_every_step") is not True or row.get("readonly_inputs_unchanged") is not True:
            raise ValueError("Missing replay integrity evidence")
        if len(row["snapshots"]) != len(program["launches"]) or row["completed_launches"] != len(program["launches"]):
            raise ValueError("Incomplete kernel chain")
        for i, snapshots in enumerate(row["snapshots"]):
            if set(snapshots) != {b["name"] for b in program["buffers"]}:
                raise ValueError("Incomplete snapshots")
            for name, snap in snapshots.items():
                checked_hash(folder / f"step{i:03d}-{name}.bin", snap["sha256"])
    program = json.loads(json.dumps(program))
    for i, audit in call_audits.items():
        program["launches"][i]["call_audit"] = audit
    return program, blobs


def prepare_case(report, row):
    task = row.get("task") or ("router" if row.get("name", "").startswith("router_") else None)
    if task is None and report.get("kind") == "swiglu_offline_simulation":
        task = "swiglu"
    sources = source_gate(report, task)
    result = {"name": row["name"], "task": task, "functional_status": row["status"],
              "dispatch": row.get("dispatch"), "comparison_group": row.get("metadata", {}).get("comparison_group"),
              "source_sha256": sources, "artifacts": row["artifacts"]}
    if row["status"] != "FUNCTIONAL_PASS":
        result.update(status="FUNCTIONAL_GATE_BLOCKED", checks=row.get("checks"))
        return result, None, None
    if "program" not in row:
        result.update(status="UNSUPPORTED_LEGACY_MANIFEST", reason="Re-export with run_candidates.py or the SIM002 Program adapter; legacy sim001 has no hashed per-step snapshots")
        return result, None, None
    program, blobs = verified_program(row)
    inputs = sorted((b["name"], b["shape"], b["dtype"], b["input_sha256"]) for b in program["buffers"] if b["readonly"])
    aux = {}
    if task == "gqa":
        aux.update(q_offsets=blobs["qoff"], kv_offsets=blobs["koff"])
    if task == "moe":
        e = next(b["shape"][0] for b in program["buffers"] if b["name"] == "wdown")
        counts = [0] * e
        k = next(b["shape"][1] for b in program["buffers"] if b["name"] == "ids")
        if any(len(set(blobs["ids"][start:start + k])) != k for start in range(0, len(blobs["ids"]), k)):
            raise ValueError("MoE requires distinct selected experts per token")
        for expert in blobs["ids"]:
            if not 0 <= expert < e:
                raise ValueError("Expert ID outside range")
            counts[expert] += 1
        aux["expert_counts"] = counts
    semantics = {"window": next((l["constants"].get("WINDOW") for l in program["launches"] if "WINDOW" in l["constants"]), None),
                 "source_contract": sources,
                 "outputs": sorted((b["name"], b["shape"], b["dtype"]) for b in program["buffers"] if b["name"] in ("output", "da", "db"))}
    result.update(input_identity=inputs, semantic_identity=semantics, evidence_verified=True,
                  known_numerical_risk=task == "swiglu" and any(b["dtype"] == "torch.float32" for b in program["buffers"]))
    if result["known_numerical_risk"]:
        result["risk_reason"] = "Same FP32 source fails the original cancellation stress cases; passing random inputs cannot establish candidate correctness. See sim003 SwiGLU audit."
    result["work"] = account(task, program, aux)
    result["launches"] = [{k: l[k] for k in ("entry", "target", "ptx_version", "grid", "constants", "options", "resources", "shared_bytes", "ptx_sha256", "cubin_sha256")} for l in program["launches"]]
    return result, program, aux


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", nargs="+", type=evidence_path, required=True)
    ap.add_argument("--spill-reports", nargs="*", type=evidence_path, default=[])
    ap.add_argument("--profile", type=Path, default=PROFILE)
    env = ap.add_mutually_exclusive_group()
    env.add_argument("--probe", type=Path, default=PROBE, help="H20 contest probe summary")
    env.add_argument("--legacy", action="store_true", help="Use the original analytical parameters")
    ap.add_argument("--scorecard", type=Path, help="Score a complete set of contest cases")
    ap.add_argument("--case", default="*", help="fnmatch on case names")
    ap.add_argument("--cache", choices=("cold", "warm"), default="cold")
    ap.add_argument("--joint-samples", type=int, default=256)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if not 16 <= args.joint_samples <= 4096:
        raise ValueError("Use 16..4096 deterministic joint sensitivity scenarios")
    profile = load_profile(args.profile)
    probe = None if args.legacy else load_probe(args.probe)
    if probe is not None:
        profile = contest_profile(profile, probe)
    design = scenarios(profile, args.joint_samples)
    spill_receipts = {}
    if args.spill_reports:
        from spill_trace import engine_hashes, load_trace
        for path in args.spill_reports:
            receipt = json.loads(path.read_text())
            if receipt["engine_sha256"] != engine_hashes() or receipt.get("gpu_execution") is not False:
                raise ValueError("Stale or non-offline spill evidence")
            for source, expected in receipt["evidence_reports"].items():
                checked_hash(evidence_path(source), expected)
            for row in receipt["cases"]:
                if row["status"] != "DYNAMIC_SPILLS_VERIFIED":
                    continue
                for launch in row["launches"]:
                    key = (evidence_path(row["source_report"]).resolve(), row["case_index"], launch["launch_index"])
                    if key in spill_receipts:
                        raise ValueError("Duplicate spill evidence")
                    spill_receipts[key] = launch
    reports = [(path, json.loads(path.read_text())) for path in args.reports]
    if all(report.get("experiment") == "sim006" for _, report in reports):
        experiment = "sim006"
    else:
        experiment = "sim005" if args.spill_reports else "sim004"
    output = {"experiment": experiment, "kind": "five_task_h20_performance_model", "h20_tested": False, "gpu_execution": False,
              "calibrated": False, "initial_cache": args.cache, "profile": profile, "profile_sha256": sha(args.profile),
              "model_sha256": {p.name: sha(p) for p in (HERE / "run_performance.py", HERE / "performance_model.py", HERE / "performance_work.py")},
              "evidence_reports": {evidence_key(p): sha(p) for p in args.reports},
              "spill_reports": {evidence_key(p): sha(p) for p in args.spill_reports},
              "scenario_design": {"seed": 1108, "joint_samples": args.joint_samples, "total": len(design),
                                  "method": "nominal, independent endpoints, paired deterministic random corners; not a probability distribution"},
              "cases": [], "comparisons": []}
    groups = defaultdict(list)
    if probe is not None:
        output["probe_sha256"] = sha(args.probe)
        output["calibration_scope"] = profile["measured_parameters"]
        output["model_sha256"]["contest_scoring.py"] = sha(HERE / "contest_scoring.py")
    for path, report in reports:
        for index, row in enumerate(report["cases"]):
            if not fnmatch.fnmatchcase(row["name"], args.case):
                continue
            identity = f"{path.name}:{index}"
            try:
                item, program, _ = prepare_case(report, row)
                if program is not None:
                    for i, launch in enumerate(program["launches"]):
                        receipt = spill_receipts.get((path.resolve(), index, i))
                        if receipt:
                            launch["dynamic_spill"] = load_trace(receipt, program, launch)
                    item["model"] = evaluate(program, item["work"], design, cache=args.cache)
                    item["status"] = item["model"]["status"]
                    if probe is not None and item["status"] == "MODELED_UNCALIBRATED":
                        item["status"] = item["model"]["status"] = "MODELED_WITH_H20_PROBE"
            except (ValueError, KeyError, OSError, TypeError) as error:
                item = {"name": row["name"], "task": row.get("task"), "functional_status": row.get("status"),
                        "comparison_group": row.get("metadata", {}).get("comparison_group"),
                        "status": "EVIDENCE_OR_MODEL_REJECTED", "reason": str(error)}
            item["id"] = identity
            item["evidence_report"] = evidence_key(path)
            output["cases"].append(item)
            if item.get("comparison_group"):
                groups[item["comparison_group"]].append(item)
    if not output["cases"]:
        raise ValueError("No cases selected")
    for group, items in groups.items():
        if len(items) == 2:
            output["comparisons"].append({"group": group, "left": items[0]["id"], "right": items[1]["id"], **compare(*items)})
        else:
            output["comparisons"].append({"group": group, "status": "RANKING_BLOCKED", "reasons": ["EXPECTED_EXACTLY_TWO_CONFIGURATIONS"]})
    for item in output["cases"]:
        item.get("model", {}).pop("_paired_times", None)
    output["summary"] = dict(Counter(row["status"] for row in output["cases"]))
    output["covered_tasks"] = sorted({row["task"] for row in output["cases"] if "model" in row})
    output["status"] = "ANALYSIS_COMPLETE_UNCALIBRATED" if not output["summary"].get("EVIDENCE_OR_MODEL_REJECTED") else "HAS_REJECTIONS"
    if probe is not None and output["status"] != "HAS_REJECTIONS":
        output["status"] = "ANALYSIS_COMPLETE_WITH_H20_PROBE"
    if args.scorecard:
        output["scorecard_sha256"] = sha(args.scorecard)
        output["submission_forecast"] = forecast_card(read_json(args.scorecard), output["cases"], profile, probe)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.spill_reports:
        output["model_sha256"].update({p.name: sha(p) for p in (HERE / "spill_service.py", HERE / "spill_trace.py", HERE / "spill_sass.py", HERE / "spill_execute.py")})
    kind = f"performance-{args.cache}" if experiment == "sim006" else "performance"
    path = args.output or report_directory(experiment) / f"results-{experiment}-{kind}-{stamp}.json"
    with path.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(path), "summary": output["summary"], "covered_tasks": output["covered_tasks"], "comparisons": output["comparisons"]}), flush=True)
    return int(output["status"] == "HAS_REJECTIONS")


if __name__ == "__main__":
    raise SystemExit(main())
