import argparse
import copy
import json
import math
import re
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import median

from performance_model import load_profile, sha, timeline


PROBE = Path(__file__).with_name("contest_probe.json")
CONTEXT = ("protocol", "family", "layout", "dtype", "input_state", "shape", "estimator")


def positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Expected a finite positive number: {value!r}")
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_probe(path=PROBE):
    probe = read_json(path)
    if probe["device"] != "NVIDIA H20" or probe["unit"] != "us" or not probe["layout"] or not probe["dtype"]:
        raise ValueError("Expected an H20 device probe in microseconds")
    for key in ("sm_count", "l2_bytes"):
        if type(probe[key]) is not int:
            raise ValueError(key)
        positive(probe[key])
    for key in ("source_sha256", "receipt_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", probe[key]):
            raise ValueError(key)
    sizes = set()
    for row in probe["samples"]:
        size = positive(row["matrix_bytes"])
        if size in sizes or type(size) is not int:
            raise ValueError("Ambiguous probe footprint")
        sizes.add(size)
        positive(row["event"])
        positive(positive(row["tiny"]) - row["event"])
        for op in ("read", "write", "copy", "warmcopy"):
            positive(positive(row[op]) - row["tiny"])
    if not sizes:
        raise ValueError("Empty probe")
    return probe


def contest_profile(profile, probe):
    result = copy.deepcopy(profile)
    launches = [s["tiny"] - s["event"] for s in probe["samples"]]
    params = result["parameters"]
    result["sources"]["contest_probe"] = {"kind": "hardware_measurement", "run_id": probe["run_id"],
                                          "receipt_sha256": probe["receipt_sha256"]}
    for name in ("sm_count", "l2_bytes"):
        params[name] = {"value": probe[name], "unit": params[name]["unit"], "source": "contest_probe"}
    params["device_launch_us"] = {"value": median(launches), "range": [min(launches), max(launches)],
                                  "unit": "us", "source": "contest_probe"}
    result["name"] += "-contest-probe"
    result["measured_parameters"] = ["sm_count", "l2_bytes", "device_launch_us"]
    return result


def traffic_time(features, params, probe):
    size = positive(features["matrix_bytes"])
    parts = features["launches"]
    if type(parts) is not int or parts < 1:
        raise ValueError("Expected a positive launch count")
    op = features["operation"]
    if op not in ("read", "write", "copy", "warmcopy"):
        raise ValueError("Unknown traffic operation")
    launch = params["device_launch_us"]
    if probe is None:
        logical_bytes = size * (2 if op in ("copy", "warmcopy") else 1)
        work = logical_bytes / (1000 * params["hbm_GBps"] * params["hbm_efficiency"])
    else:
        if features["layout"] != probe["layout"]:
            raise ValueError("No matched traffic layout")
        rows = [s for s in probe["samples"] if s["matrix_bytes"] == size]
        if len(rows) != 1:
            raise ValueError("No matched traffic footprint")
        row = rows[0]
        launch = row["tiny"] - row["event"]
        work = row[op] - row["tiny"]
    return timeline([work / parts + launch] * parts, 0, params)["e2e_synced_us"]


def context_key(context):
    values = tuple(context[name] for name in CONTEXT)
    if not all(isinstance(v, str) and v for v in values):
        raise ValueError("Incomplete calibration context")
    return values


def fit_offsets(samples):
    # 同一源码的复测保留波动幅度，每份源码对校准中位数有相同权重。
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for sample in samples:
        context = context_key(sample["context"])
        source = sample["source_sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", source) or not sample["receipt_id"]:
            raise ValueError("Unbound calibration observation")
        identity = (context, sample["receipt_id"])
        if identity in seen:
            raise ValueError("Repeated calibration observation")
        seen.add(identity)
        groups[context][source].append(positive(sample["official_us"]) - positive(sample["estimate_us"]))
    result = {}
    for context, by_source in groups.items():
        centers = {source: median(values) for source, values in by_source.items()}
        error = None
        if len(centers) >= 2:
            errors = []
            for source, values in by_source.items():
                other = median(center for key, center in centers.items() if key != source)
                errors.extend(abs(v - other) for v in values)
            error = max(errors)
        result[context] = (median(centers.values()), error, frozenset(centers))
    return result


def points(value):
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def score_times(rule, cases, times):
    if not times or len(times) != len(cases) or not all(math.isfinite(t) and t >= 0 for t in times):
        raise ValueError("Invalid timing vector")
    base = Decimal(str(positive(rule["base_points"])))
    durations = [Decimal(str(t)) for t in times]
    cap = Decimal("1.5")
    if rule["mode"] == "practice_total":
        benchmark = Decimal(str(positive(rule["benchmark_total_us"])))
        total = sum(durations)
        return points(base * (cap if total == 0 else min(cap, benchmark / total)))
    if rule["mode"] != "preliminary_per_case":
        raise ValueError("Unknown contest scoring formula")
    benchmarks = [Decimal(str(positive(c["benchmark_us"]))) for c in cases]
    terms = [cap if t == 0 else min(cap, b / t) for b, t in zip(benchmarks, durations, strict=True)]
    return points(base * sum(terms) / len(terms))


def score_submission(rule, cases, incumbent, offsets, candidate_sha256):
    if not re.fullmatch(r"[0-9a-f]{64}", candidate_sha256):
        raise ValueError("Invalid candidate source digest")
    if (not cases or len({c["id"] for c in cases}) != len(cases)
            or [c["id"] for c in cases] != rule["case_ids"]):
        raise ValueError("Missing or duplicate scoring case")
    if not math.isfinite(incumbent) or not 0 <= incumbent <= positive(rule["base_points"]) * 1.5:
        raise ValueError("Invalid incumbent score")
    if rule["mode"] not in ("practice_total", "preliminary_per_case"):
        raise ValueError("Unknown contest scoring formula")
    awaiting = rule["mode"] == "preliminary_per_case" and any(c.get("benchmark_us") is None for c in cases)
    if rule["mode"] == "preliminary_per_case" and any("incumbent_us" in c for c in cases):
        if awaiting or not all("incumbent_us" in c for c in cases):
            raise ValueError("Incumbent contributions require a complete scorecard")
        if score_times(rule, cases, [positive(c["incumbent_us"]) for c in cases]) != incumbent:
            raise ValueError("Incumbent case times disagree with incumbent score")
    rows = []
    for case in cases:
        raw = positive(case["estimate_us"])
        offset, error, sources = offsets.get(context_key(case["context"]), (0.0, None, frozenset()))
        if candidate_sha256 in sources:
            raise ValueError("Candidate also used for calibration")
        estimate = positive(raw + offset)
        row = {"id": case["id"], "context": case["context"], "raw_us": raw, "estimate_us": estimate,
               "offset_us": offset, "validation_error_us": error, "calibration_sources": len(sources)}
        if rule["mode"] == "preliminary_per_case" and case.get("benchmark_us") is not None:
            baseline = positive(case["benchmark_us"])
            row["benchmark_us"] = baseline
            row["score_contribution"] = rule["base_points"] * min(1.5, baseline / estimate) / len(cases)
            if "incumbent_us" in case:
                old = rule["base_points"] * min(1.5, baseline / positive(case["incumbent_us"])) / len(cases)
                row["contribution_change"] = row["score_contribution"] - old
        rows.append(row)
    if awaiting:
        return {"rule": rule, "candidate_sha256": candidate_sha256, "cases": rows, "incumbent": incumbent,
                "score": None, "score_change": None, "score_low": None, "score_high": None,
                "decision": "AWAITING_BENCHMARKS"}
    estimate = score_times(rule, cases, [r["estimate_us"] for r in rows])
    low = high = None
    decision = "UNRESOLVED"
    if all(r["validation_error_us"] is not None for r in rows):
        slow = [r["estimate_us"] + r["validation_error_us"] for r in rows]
        fast = [max(0, r["estimate_us"] - r["validation_error_us"]) for r in rows]
        low, high = score_times(rule, cases, slow), score_times(rule, cases, fast)
        if high <= incumbent:
            decision = "SKIP"
        elif low > incumbent:
            decision = "PRIORITIZE"
    return {"rule": rule, "candidate_sha256": candidate_sha256, "cases": rows, "score": estimate,
            "incumbent": incumbent, "score_change": estimate - incumbent, "score_low": low, "score_high": high,
            "decision": decision}


def forecast_card(card, results, profile, probe):
    if any(s["source_sha256"] == card["candidate_sha256"] for s in card["calibration"]):
        raise ValueError("Candidate also used for calibration")
    index = {r["id"]: r for r in results}
    model_ids = [c["model_id"] for c in card["cases"] if c["context"]["estimator"] == "replay_e2e"]
    if len(index) != len(results) or len(model_ids) != len(set(model_ids)):
        raise ValueError("Ambiguous replay identity")
    rows = []
    missing = []
    declared = [c["id"] for c in card["cases"]]
    if not declared or len(set(declared)) != len(declared) or declared != card["rule"]["case_ids"]:
        raise ValueError("Incomplete case list")
    for case in card["cases"]:
        context = case["context"]
        context_key(context)
        if context["protocol"] != card["protocol"]:
            raise ValueError("Inconsistent timing protocol")
        kind = context["estimator"]
        if kind == "replay_e2e":
            item = index[case["model_id"]]
            if (not item.get("evidence_verified") or item.get("functional_status") != "FUNCTIONAL_PASS"
                    or item.get("known_numerical_risk") or not item["model"]["nominal"]["ranking_eligible"]
                    or context["family"] != item["task"] or context["input_state"] != item["model"]["cache"]
                    or case["input_identity"] != [list(item_id) for item_id in item["input_identity"]]
                    or card["candidate_sha256"] not in [v for k, v in item["source_sha256"].items() if k.endswith("/solution.py")]):
                raise ValueError("Replay evidence ineligible for scoring")
            value = item["model"]["nominal"]["timeline"]["e2e_synced_us"]
            if "call_timing" in case:
                value = replay_call_time(item, case["call_timing"], profile, card["protocol"])
        elif kind == "unavailable":
            if not case.get("reason"):
                raise ValueError("Missing evidence reason")
            missing.append({"id": case["id"], "reason": case["reason"]})
            continue
        elif kind == "traffic":
            feature = case["traffic"]
            if feature["layout"] != context["layout"]:
                raise ValueError("Incompatible memory probe context")
            if probe is not None and (context["dtype"] != probe["dtype"] or probe["layout"] != feature["layout"]
                                      or feature["matrix_bytes"] not in {s["matrix_bytes"] for s in probe["samples"]}):
                missing.append({"id": case["id"], "reason": "NO_MATCHING_TRAFFIC_PROBE"})
                continue
            state = "repeated" if feature["operation"] == "warmcopy" else "rotated"
            if context["input_state"] != state:
                raise ValueError("Incompatible probe input state")
            params = {k: p["value"] for k, p in profile["parameters"].items()}
            value = traffic_time(feature, params, probe)
        elif kind == "paired_measurement":
            value = positive(case["measurement_us"])
        else:
            raise ValueError("Unknown timing estimator")
        rows.append({**case, "estimate_us": value})
    if missing:
        return {"rule": card["rule"], "candidate_sha256": card["candidate_sha256"], "incumbent": card["incumbent"],
                "cases": rows, "missing_cases": missing, "score": None, "score_change": None,
                "score_low": None, "score_high": None, "decision": "UNRESOLVED"}
    return score_submission(card["rule"], rows, card["incumbent"], fit_offsets(card["calibration"]),
                            card["candidate_sha256"])


def replay_call_time(item, measurement, profile, protocol):
    if protocol != measurement["protocol"] or not measurement["receipt_id"]:
        raise ValueError("Unbound call measurements")
    nominal = item["model"]["nominal"]
    params = {k: p["value"] for k, p in profile["parameters"].items()}
    for k, t in measurement.get("host_costs", {}).items():
        if (k not in ("host_wrapper_us", "host_submit_us", "host_sync_us", "host_allocation_us")
                or type(t) not in (int, float) or not math.isfinite(t) or t < 0):
            raise ValueError("Invalid measured host cost")
        params[k] = t
    steps, next_kernel = [], 0
    for step in measurement["device_steps"]:
        if set(step) == {"clear_us"}:
            steps.append(positive(step["clear_us"]))
            continue
        if (type(step.get("kernel")) is not int or step["kernel"] != next_kernel
                or next_kernel >= len(nominal["kernels"])):
            raise ValueError("Unknown step or incorrect kernel order")
        if set(step) == {"kernel"}:
            duration = nominal["kernels"][next_kernel]["device_us"]
        elif set(step) == {"kernel", "cubin_sha256", "device_us"}:
            inputs = [list(identity) for identity in item["input_identity"]]
            if (step["cubin_sha256"] != item["launches"][next_kernel]["cubin_sha256"]
                    or measurement.get("input_identity") != inputs):
                raise ValueError("Kernel timing does not match replay binary and inputs")
            duration = positive(step["device_us"])
        else:
            raise ValueError("Unknown kernel timing fields")
        steps.append(duration)
        next_kernel += 1
    if next_kernel != len(nominal["kernels"]):
        raise ValueError("Incomplete call")
    return timeline(steps, nominal["allocations"], params)["e2e_synced_us"]


def main():
    parser = argparse.ArgumentParser(description="Score a contest submission from measured cases or a replay report")
    parser.add_argument("card", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--probe", type=Path, default=PROBE)
    args = parser.parse_args()
    probe = load_probe(args.probe)
    report = read_json(args.report) if args.report else {"cases": []}
    if args.report and report.get("probe_sha256") != sha(args.probe):
        raise ValueError("Replay report and scorecard require the same contest probe")
    profile = report["profile"] if args.report else contest_profile(load_profile(), probe)
    result = forecast_card(read_json(args.card), report["cases"], profile, probe)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
