"""Uncalibrated H20 analytical model; never consumes H100 cycle estimates."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re

from performance_work import TASKS, ceildiv

PROFILE = Path(__file__).with_name("h20_profile.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_profile(path=PROFILE):
    p = json.loads(Path(path).read_text())
    if p["schema"] != "SIM004_PROFILE_V1" or set(p["scope"]) != TASKS or p["calibrated"] is not False:
        raise ValueError("This model requires the five-task uncalibrated profile")
    for name, value in p["parameters"].items():
        if value["source"] not in p["sources"] or not value["unit"]:
            raise ValueError(f"Missing parameter provenance: {name}")
        ends = value.get("range", [value["value"], value["value"]])
        if len(ends) != 2 or any(not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0 for x in [value["value"], *ends]):
            raise ValueError(f"Invalid numeric parameter: {name}")
        if not ends[0] <= value["value"] <= ends[1]:
            raise ValueError(f"Nominal value outside range: {name}")
        if value["unit"] == "fraction" and ends[1] > 1:
            raise ValueError(f"Fraction outside [0,1]: {name}")
        if value["unit"] == "policy" and any(x not in (0, 1) for x in [value["value"], *ends]):
            raise ValueError(f"Unknown discrete policy: {name}")
        if (name.endswith(("Gops", "GBps", "TFps", "warps")) or name in ("clock_GHz", "sm_count", "spill_warp_issue_per_cycle_sm") or "efficiency" in name) and ends[0] <= 0:
            raise ValueError(f"Nonpositive rate/capacity: {name}")
    return p


def scenarios(profile, samples=256):
    """Nominal + one-at-a-time endpoints + deterministic joint corner sampling.

    This is a sensitivity design, not a probability distribution or exhaustive
    interval proof. Identical scenarios are used for both sides of a comparison.
    """
    base = {k: v["value"] for k, v in profile["parameters"].items()}
    varying = {k: v["range"] for k, v in profile["parameters"].items() if "range" in v}
    rows = [("nominal", base)]
    for key, ends in varying.items():
        for side, value in zip(("low", "high"), ends):
            rows.append((f"{key}:{side}", {**base, key: value}))
    rng = random.Random(1108)
    for i in range(samples):
        rows.append((f"joint:{i}", {**base, **{key: rng.choice(ends) for key, ends in varying.items()}}))
    return rows


def residency(launch, p):
    threads = launch["threads"]
    if threads <= 0 or threads > 1024 or threads % 32:
        raise ValueError("Invalid block size")
    warps = threads // 32
    reg = launch["resources"]["reg"]
    if not 0 < reg <= 255:
        raise ValueError("Invalid measured register count")
    allocated_regs = ceildiv(reg * 32, p["register_granule_per_warp"]) * p["register_granule_per_warp"] * warps
    user_shared = launch["shared_bytes"] + launch["resources"].get("shared", 0)
    if user_shared > p["shared_per_block"]:
        raise ValueError("Per-block shared memory exceeds modeled device capacity")
    allocated_shared = ceildiv(user_shared + p["reserved_shared_per_block"], p["shared_granule"]) * p["shared_granule"]
    limits = {"registers": p["registers_per_sm"] // allocated_regs,
              "shared": p["shared_per_sm"] // allocated_shared,
              "warps": p["warps_per_sm"] // warps,
              "blocks": p["blocks_per_sm"]}
    resident = int(min(limits.values()))
    if resident < 1:
        raise ValueError("Launch cannot reside on modeled H20")
    return {"ctas_per_sm": resident, "warps_per_sm": resident * warps,
            "occupancy_fraction_upper_bound": resident * warps / p["warps_per_sm"],
            "allocated_registers_per_cta": allocated_regs, "allocated_shared_per_cta": allocated_shared,
            "limiting_resources": [k for k, v in limits.items() if v == resident],
            "limits": limits, "status": "ARCHITECTURAL_RESOURCE_BOUND_NOT_MEASURED_OCCUPANCY"}


def utilization(work, launch, occupancy, p, saturation):
    if not work["active_ctas"]:
        return 1.0
    need = max(1, math.ceil(saturation / (launch["threads"] / 32)))
    width = min(occupancy["ctas_per_sm"], need)
    capacity = p["sm_count"] * width
    waves = math.ceil(work["active_ctas"] / capacity)
    grid_fraction = work["active_ctas"] / (waves * capacity)
    return grid_fraction * min(1.0, occupancy["warps_per_sm"] / saturation)


def spill_bounds(launch, work):
    if "dynamic_spill" in launch:
        trace = launch["dynamic_spill"]
        if trace.get("status") != "COMPLETE_DYNAMIC_LOCAL_ACCESSES" or not trace.get("trace_sha256"):
            raise ValueError("Unverified dynamic spill attachment")
        total = trace["load_bytes"] + trace["store_bytes"]
        return {"status": trace["status"], "static_sites": len(trace["static_sites"]),
                "dynamic_bytes_min": total, "dynamic_bytes_cap": total,
                "load_bytes": trace["load_bytes"], "store_bytes": trace["store_bytes"],
                "trace_sha256": trace["trace_sha256"], "ranking_eligible": True,
                "method": "Verified SASS control/address projection plus independent source/static-site count"}
    sites = launch.get("sass_stack_or_call_instructions", [])
    stack = launch["resources"].get("stack", 0)
    local = launch["resources"].get("local", 0)
    load, store, count = 0, 0, 0
    for line in sites:
        if re.search(r"\bCALL\b", line) and not launch.get("call_audit", {}).get("acyclic_helpers"):
            raise ValueError("Unmodeled SASS call path; no spill upper bound")
        found = re.search(r"\b(LDL|STL)((?:\.[A-Za-z0-9]+)*)\s", line)
        if not found:
            continue
        widths = [int(x) for x in re.findall(r"\.(?:U|S)?(8|16|32|64|128)(?:\.|$)", found[2])]
        width = widths[0] // 8 if widths else 4
        if found[1] == "LDL":
            load += width
        else:
            store += width
        count += 1
    if (stack or local) and not count:
        raise ValueError("Stack/local allocation without audited LDL/STL sites")
    # A callee can be shared by multiple call sites. The audited helper bodies
    # have no cycles/nested calls; each call site is covered by the source-loop
    # bound. Count all sites even when the fast FP64 divide skips the helper.
    call_factor = max(1, launch.get("call_audit", {}).get("call_sites", 0))
    multiplier = launch["threads"] * work["launched_ctas"] * (work["max_loop_visits"] + 1) * call_factor
    return {"status": "BOUNDED_UNTRACED_SPILLS" if count else "NO_REPORTED_SPILLS",
            "static_sites": count, "static_load_bytes_per_thread": load, "static_store_bytes_per_thread": store,
            "dynamic_bytes_min": 0, "dynamic_bytes_cap": (load + store) * multiplier,
            "call_audit": launch.get("call_audit"),
            "method": "Every static local-memory site, all lanes and launched CTAs, (max source loop visits + 1) * max(1, audited call sites). Helpers must be acyclic. Predicates/branches untraced; zero lower bound. Not a cycle or SASS execution model.",
            "ranking_eligible": not count}


def audit_calls(launch, sass):
    """Accept only the observed GQA FP64 division's forward-only local helper.

    This is a bounded control-flow audit, not a general SASS interpreter.
    Unknown indirect jumps, backward edges, nested calls and missing returns
    fail closed. The caller's loop count still comes from source accounting.
    """
    insns = {}
    for line in sass.splitlines():
        m = re.search(r"/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*;", line)
        if m:
            pc = int(m[1], 16)
            if pc in insns:
                raise ValueError("Multiple SASS functions need a separate call audit")
            insns[pc] = m[2]
    calls = [s for s in insns.values() if re.search(r"\bCALL\b", s)]
    if not calls:
        return {"call_sites": 0, "acyclic_helpers": True, "targets": []}
    if launch["entry"] != "_gqa":
        raise ValueError("Device calls outside the reviewed GQA division path")
    targets = []
    for call in calls:
        m = re.fullmatch(r"CALL\.REL\.NOINC\s+0x([0-9a-f]+)", call)
        if not m:
            raise ValueError("Unsupported device-call encoding")
        target = int(m[1], 16)
        returns = [pc for pc, s in insns.items() if pc >= target and s.startswith("RET.REL.NODEC")]
        if target not in insns or not returns:
            raise ValueError("Missing division helper target or return")
        end = min(returns)
        for pc, text in insns.items():
            if not target <= pc <= end:
                continue
            if re.search(r"\b(?:CALL|JMP|BRX|JMX|EXIT)\b", text):
                raise ValueError("Unsupported helper control flow")
            if re.search(r"\bBRA\b", text):
                branch = re.search(r"\bBRA(?:\s+P\d+,)?\s+0x([0-9a-f]+)$", text)
                if not branch or not pc < int(branch[1], 16) <= end:
                    raise ValueError("Non-acyclic division helper")
        targets.append({"entry_pc": target, "return_pc": end})
    return {"call_sites": len(calls), "acyclic_helpers": True, "targets": targets,
            "sass_sha256": hashlib.sha256(sass.encode()).hexdigest(),
            "scope": "GQA compiled FP64 division helper, all forward branches; invocation predicates untraced"}


def kernel_time(launch, work, p, *, cache="cold"):
    occ = residency(launch, p)
    spill = spill_bounds(launch, work)
    um = utilization(work, launch, occ, p, p["memory_saturation_warps"])
    uf = utilization(work, launch, occ, p, p["scalar_saturation_warps"])
    ut = utilization(work, launch, occ, p, p["tensor_saturation_warps"])
    footprint = sum(v["unique_read_bytes"] + v["write_bytes"] for v in work["traffic"].values())
    capacity = min(1.0, p["l2_bytes"] / max(1, footprint))
    reads = sum(v["read_bytes"] for v in work["traffic"].values())
    writes = sum(v["write_bytes"] for v in work["traffic"].values())
    hbm = writes
    for v in work["traffic"].values():
        first_hit = p["intermediate_cache_hit"] if v["intermediate"] else p["warm_input_cache_hit"] if cache == "warm" else 0
        hbm += v["unique_read_bytes"] * (1 - first_hit * capacity)
        hbm += (v["read_bytes"] - v["unique_read_bytes"]) * (1 - p["repeat_cache_hit"] * capacity)
    spill_service = None
    if "dynamic_spill" in launch:
        from spill_service import service
        spill_service = service(launch["dynamic_spill"], launch, occ, p)
        spill_bytes = spill["dynamic_bytes_cap"]
        spill_l2, spill_hbm = spill_service["l2_bytes"], spill_service["hbm_bytes"]
    else:
        spill_bytes = spill["dynamic_bytes_cap"] * p["spill_executed_fraction"]
        spill_l2 = spill_bytes * (1 - p["spill_l1_hit"])
        spill_hbm = spill_l2 * (1 - p["repeat_cache_hit"] * capacity)
    amp, aux = p["transaction_amplification"], p["aux_work_multiplier"]
    ops = work["operations"]
    terms = {
        "hbm": (hbm * amp + spill_hbm * (1 if spill_service else amp)) / (p["hbm_GBps"] * p["hbm_efficiency"] * 1000 * um),
        "l2": ((reads + writes) * amp + spill_l2 * (1 if spill_service else amp)) / (p["l2_GBps"] * 1000 * um),
        "shared_l1_proxy": (work["shared_bytes_proxy"] * aux + (spill_service["l1_sector_bytes"] if spill_service else spill_bytes)) / (p["shared_GBps"] * 1000 * um),
        "tensor": ops["tensor"] / (p["sm_count"] * p["clock_GHz"] * 1000 * p["tensor_ops_per_cycle_sm"] * p["tensor_efficiency"] * ut),
        "fp32": ops["fp32"] / (p["fp32_TFps"] * 1e6 * p["scalar_efficiency"] * uf),
        "fp64": ops["fp64"] / (p["fp64_TFps"] * 1e6 * p["scalar_efficiency"] * uf),
    }
    for kind in ("exp2", "div32", "div64", "convert", "integer", "atomic"):
        rate = (p["sm_count"] * p["clock_GHz"] * p["integer_ops_per_cycle_sm"] * p["integer_efficiency"]
                if kind == "integer" else p[kind + "_Gops"])
        terms[kind] = ops[kind] * (aux if kind in ("convert", "integer") else 1) / (rate * 1000 * uf)
    terms["atomic"] = max(terms["atomic"], work["max_atomic_chain"] * p["atomic_hotspot_ns"] / 1000)
    waves = math.ceil(work["launched_ctas"] / (p["sm_count"] * occ["ctas_per_sm"]))
    # A requested compiler stage count is not evidence of compiled overlap.
    # The explicit dependency_exposed sensitivity represents unknown overlap.
    stages = 1
    critical = work["max_loop_visits"] * p["memory_latency_ns"] / 1000 / max(1, stages)
    critical += (p["tensor_latency_cycles"] if ops["tensor"] else 0) / (p["clock_GHz"] * 1000)
    critical += work["reduction_depth_proxy"] * p["reduction_latency_cycles"] / (p["clock_GHz"] * 1000)
    terms["dependency_proxy"] = critical * p["dependency_exposed"] * waves
    if spill_service:
        terms["local_issue"] = spill_service["instruction_count"] / (p["sm_count"] * p["clock_GHz"] * 1000 * p["spill_warp_issue_per_cycle_sm"] * um)
        max_loads = max(sum(op == "LDL" and active for _, op, _, _, active in g["events"]) for g in launch["dynamic_spill"]["classes"])
        hit, l2_hit = spill_service["load_hit_fraction"], spill_service["effective_l2_hit_assumption"]
        cycles = hit*p["spill_load_latency_cycles"] + (1-hit)*(l2_hit*p["spill_l2_latency_cycles"] + (1-l2_hit)*p["memory_latency_ns"]*p["clock_GHz"])
        spill_service["weighted_load_latency_cycles_proxy"] = cycles
        terms["local_dependency_proxy"] = max_loads * waves * cycles * p["spill_dependency_exposed"] / (p["clock_GHz"] * 1000)
    dominant = max(terms, key=terms.get)
    largest = terms[dominant]
    # Independent-pipeline overlap is unknown. The explicit serial fraction
    # interpolates max(pipelines) and sum(pipelines), not a fitted residual.
    service = largest + p["serialization_fraction"] * (sum(terms.values()) - largest)
    dispatch = work["launched_ctas"] * p["cta_schedule_cycles"] / (p["sm_count"] * p["clock_GHz"] * 1000)
    return {"device_us": p["device_launch_us"] + dispatch + service,
            "pipeline_us": terms, "dominant": dominant, "cta_dispatch_us": dispatch,
            "device_launch_us": p["device_launch_us"], "occupancy": occ, "resident_waves": waves,
            "utilization_assumptions": {"memory": um, "scalar": uf, "tensor": ut},
            "modeled_hbm_bytes": hbm*amp + spill_hbm*(1 if spill_service else amp),
            "modeled_l2_bytes": (reads+writes)*amp + spill_l2*(1 if spill_service else amp),
            "modeled_spill_bytes": spill_bytes, "spill": spill, "spill_service": spill_service}


def timeline(durations, allocations, p):
    """One stream, async CPU submission, allocations before first launch."""
    host = p["host_wrapper_us"] + allocations * p["host_allocation_us"]
    gpu, events = 0.0, []
    for i, duration in enumerate(durations):
        host += p["host_submit_us"]
        start = max(host, gpu)
        gpu = start + duration
        events.append({"kernel": i, "submitted_us": host, "start_us": start, "end_us": gpu})
    return {"device_chain_us": sum(durations), "e2e_synced_us": max(host, gpu) + (p["host_sync_us"] if durations else 0),
            "graph_replay_us": p["graph_submit_us"] + sum(durations) if durations else 0,
            "host_prelaunch_us": p["host_wrapper_us"] + allocations * p["host_allocation_us"],
            "events": events,
            "boundary": "warm JIT/caching allocator; resident inputs; one stream; eager E2E includes final wait. Graph replay excludes capture, allocations and final host wait. Official timing boundary unconfirmed."}


def predict(program, works, p, *, cache="cold"):
    if cache not in ("cold", "warm") or len(works) != len(program["launches"]):
        raise ValueError("Invalid cache state or launch accounting")
    memory = sum(b["bytes"] for b in program["buffers"])
    if memory > p["memory_bytes"]:
        raise ValueError("Buffers exceed modeled H20 capacity")
    kernels = [kernel_time(l, w, p, cache=cache) for l, w in zip(program["launches"], works)]
    allocations = sum(not b["readonly"] for b in program["buffers"])
    return {"kernels": kernels, "timeline": timeline([k["device_us"] for k in kernels], allocations, p),
            "allocated_bytes": memory, "allocations": allocations,
            "ranking_eligible": all(k["spill"]["ranking_eligible"] for k in kernels)}


def evaluate(program, works, design, *, cache="cold"):
    predictions = [predict(program, works, params, cache=cache) for _, params in design]
    metrics = ("device_chain_us", "e2e_synced_us", "graph_replay_us")
    bounds = {m: {"min": min(p["timeline"][m] for p in predictions), "nominal": predictions[0]["timeline"][m],
                  "max": max(p["timeline"][m] for p in predictions)} for m in metrics}
    sensitivity = []
    for (name, _), prediction in zip(design[1:], predictions[1:]):
        if name.startswith("joint:"):
            continue
        sensitivity.append({"parameter_endpoint": name, "e2e_synced_us": prediction["timeline"]["e2e_synced_us"],
                            "relative_change": prediction["timeline"]["e2e_synced_us"] / max(1e-20, bounds["e2e_synced_us"]["nominal"]) - 1})
    dominant = [dict(Counter(p["kernels"][i]["dominant"] for p in predictions)) for i in range(len(works))]
    return {"status": "MODELED_UNCALIBRATED" if predictions[0]["ranking_eligible"] else "SCENARIO_ONLY_UNTRACED_SPILLS",
            "cache": cache, "nominal": predictions[0], "scenario_ranges_us": bounds,
            "scenario_count": len(design), "range_semantics": "sampled assumptions, not confidence intervals or guaranteed bounds",
            "dominant_pipeline_counts": dominant,
            "top_sensitivities": sorted(sensitivity, key=lambda x: abs(x["relative_change"]), reverse=True)[:8],
            "_paired_times": {m: [p["timeline"][m] for p in predictions] for m in metrics}}


def compare(left, right, *, metric="e2e_synced_us"):
    reasons = []
    if left.get("task") != right.get("task") or not left.get("input_identity") or left.get("input_identity") != right.get("input_identity") or left.get("semantic_identity") != right.get("semantic_identity"):
        reasons.append("DIFFERENT_INPUT_OR_SEMANTICS")
    for item in (left, right):
        if item.get("functional_status") != "FUNCTIONAL_PASS":
            reasons.append("FUNCTIONAL_GATE_FAILED")
        if item.get("known_numerical_risk"):
            reasons.append("KNOWN_SAME_SOURCE_NUMERICAL_FAILURE")
        if not item.get("model", {}).get("nominal", {}).get("ranking_eligible", False):
            reasons.append("UNTRACED_SPILLS_OR_MISSING_MODEL")
    if reasons:
        return {"status": "RANKING_BLOCKED", "reasons": sorted(set(reasons))}
    a, b = left["model"]["_paired_times"][metric], right["model"]["_paired_times"][metric]
    if len(a) != len(b) or not a or any(x <= 0 for x in [*a, *b]):
        raise ValueError("Comparison scenario mismatch")
    ratios = [x / y for x, y in zip(a, b)]
    if min(ratios) > 1.02:
        decision = "RIGHT_FASTER_IN_SAMPLED_SCENARIOS"
    elif max(ratios) < 1 / 1.02:
        decision = "LEFT_FASTER_IN_SAMPLED_SCENARIOS"
    else:
        decision = "INCONCLUSIVE_WITHIN_MODEL_ASSUMPTIONS"
    return {"status": decision, "metric": metric, "left_over_right_model_ratio": {"min": min(ratios), "nominal": ratios[0], "max": max(ratios)},
            "paired_scenarios": len(ratios), "margin": 0.02,
            "limitation": "Model screening on these validated inputs only. Not measured H20 speedup or proof over the full parameter interval."}
