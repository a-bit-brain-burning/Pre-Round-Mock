"""Cubin/input-bound dynamic local access records for the audited contest kernels."""

from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess

from spill_execute import Lane, abi_parameters
from spill_sass import Projection, UnsupportedSpill, parse, target, width

HERE = Path(__file__).resolve().parent
ENGINE = ("spill_sass.py", "spill_execute.py", "spill_trace.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def engine_hashes():
    return {name: digest(HERE/name) for name in ENGINE}


def binding(program, launch):
    return canonical({"launch": {k: v for k, v in launch.items() if k not in ("call_audit", "dynamic_spill")},
                      "buffers": program["buffers"], "protocol": program["protocol"]})


def sink_closed_form(q, launch, cta, sites):
    """Check the two paths around the source KV loop against the audited cubin layout."""
    c = launch["constants"]
    shape = (c["LQ"], c["LK"], c["HQ"], c["HK"], c["D"], c["BD"], c["BM"], c["BN"], c["WINDOW"], c["P_PARTS"])
    if shape != (8, 65, 4, 1, 128, 128, 16, 32, 0, 3) or launch["grid"] != [2, 1, 1]:
        raise UnsupportedSpill("Unaudited Sink tiled configuration")
    branches = [q.insns[pc] for pc in q.selected if q.insns[pc].base == "BRA"]
    if len(branches) != 1 or not branches[0].predicate:
        raise UnsupportedSpill("Expected one live Sink loop-entry branch")
    branch = branches[0]
    join = q.ipdom[q.block_of[branch.pc]]
    # Four prologue words, one word on either arm, then five epilogue reads.
    offsets = (16, 12, 8, 0, 4, 4, 4, 16, 0, 12, 8)
    signature = [(i.base, i.local_offset(), width(i.op)) for i in sites]
    expected = [("STL" if n < 6 else "LDL", offset, 4) for n, offset in enumerate(offsets)]
    if signature != expected or join is None or not (
        sites[3].pc < branch.pc < sites[4].pc < target(branch) <= sites[5].pc < join <= sites[6].pc
    ):
        raise UnsupportedSpill("Unexpected Sink frame offsets or branch layout")
    for n, site in enumerate(sites):
        controls = {branch.pc} if n in (4, 5) else set()
        if q.control[q.block_of[site.pc]] != controls:
            raise UnsupportedSpill("Unexpected control dependency for a Sink spill site")
    group = c["HQ"] // c["HK"]
    last_q = min(c["LQ"] - 1, ((cta[0] + 1) * c["BM"] - 1) // group)
    end_key = c["LK"] - c["LQ"] + last_q + 1
    kv_visits = (end_key + c["BN"] - 1) // c["BN"]
    chosen = sites[5] if kv_visits else sites[4]
    return {"load_bytes_per_lane": 20, "store_bytes_per_lane": 20, "loop_visits": 0,
            "loop_branch_pc": None, "source_kv_visits": kv_visits,
            "expected_site_pcs": [i.pc for i in (*sites[:4], chosen, *sites[6:])],
            "expected_branches": [[branch.pc, int(kv_visits > 0)]],
            "method": "Source KV loop entry and independent prologue/branch/epilogue frame-word audit"}


def check_sink_path(oracle, events, branches):
    active_sites = [pc for pc, _, _, _, active in events if active]
    if active_sites != oracle["expected_site_pcs"] or branches != oracle["expected_branches"]:
        raise UnsupportedSpill("Independent Sink spill path or site order mismatch")


def closed_form(q, launch, buffers, cta):
    """Independent source loop count times static sites in audited regions.

    The slice executor does not use this count. Refuse nested/predicated local
    topology; all accepted static sites execute once per region visit.
    """
    locals_ = [q.insns[pc] for pc in sorted(q.locals)]
    if any(i.predicate for i in locals_):
        raise UnsupportedSpill("Closed-form oracle does not cover predicated local sites")
    loops = [i for pc, i in q.insns.items() if pc in q.reachable and i.base == "BRA" and target(i) < pc
             and any(target(i) <= p <= pc for p in q.locals)]
    if launch["entry"] == "_sink_tiled":
        if loops:
            raise UnsupportedSpill("Unexpected Sink local access inside a backward branch")
        return sink_closed_form(q, launch, cta, locals_)
    if len(loops) > 1:
        raise UnsupportedSpill("Closed-form oracle needs a single local-memory loop")
    if launch["entry"] == "_packed_gemm":
        if len(loops) != 1:
            raise UnsupportedSpill("Unrecognized Router spill loop")
        visits = 3072 // (launch["constants"]["BK"] * launch["constants"]["SPLIT"])
        selected = locals_
        position = None
    elif launch["entry"] == "_gqa":
        def offsets(name):
            data = buffers[name][1]
            return struct.unpack("<" + "i"*(len(data)//4), data)
        qo, ko = offsets("qoff"), offsets("koff")
        query = cta[0]
        batch = next((b for b in range(len(qo)-1) if qo[b] <= query < qo[b+1]), None)
        if batch is None:
            raise UnsupportedSpill("Query outside packed offsets")
        position = ko[batch+1]-ko[batch] - qo[batch+1] + query
        bn = 32 if position < 32 else launch["constants"]["BN"]
        win = launch["constants"]["WINDOW"]
        first = (max(0, position-win+1) if win else 0) // bn * bn
        visits = (position+1-first+bn-1)//bn
        # Identify the actual source position split, then validate its topology.
        selectors = []
        for pc in sorted(q.selected):
            inst = q.insns[pc]
            if inst.op == "ISETP.GT.AND" and inst.args[3] == "0x1f":
                following = q.follow[pc]
                while following is not None:
                    branch = q.insns[following]
                    if branch.base == "BRA":
                        if branch.predicate == inst.args[0]:
                            selectors.append(branch)
                        break
                    if inst.args[0] in branch.definitions() or branch.base in ("EXIT", "BREAK", "CALL"):
                        break
                    following = q.follow[following]
        if len(selectors) != 1 or len(loops) != 1:
            raise UnsupportedSpill("Unrecognized GQA position/loop topology")
        split, loop = selectors[0], loops[0]
        large, start, end = target(split), target(loop), loop.pc
        if not split.pc < large <= start < end:
            raise UnsupportedSpill("Unrecognized GQA spill region order")
        if position < 32:
            selected = [i for i in locals_ if i.pc < large or i.pc > end]
        else:
            selected = [i for i in locals_ if i.pc < split.pc or i.pc >= large]
        # No local event may depend on any numerical branch or extra loop.
        allowed = {split.pc, loop.pc}
        for inst in selected:
            for pc in q.control[q.block_of[inst.pc]]:
                branch = q.insns[pc]
                if pc in allowed:
                    continue
                # Compiler emits zero-trip guards. Contract position>=0 ensures
                # these are fallthrough; the source-based byte check catches skips.
                if branch.base != "BRA" or target(branch) <= pc or target(branch) <= end:
                    raise UnsupportedSpill("Extra local-region control outside count oracle")
    else:
        raise UnsupportedSpill("No dynamic spill count oracle for this kernel")
    counts = Counter()
    loop = loops[0] if loops else None
    for inst in selected:
        times = visits if loop and target(loop) <= inst.pc <= loop.pc else 1
        # For GQA's small branch the large loop is never entered.
        counts[inst.base] += width(inst.op)*times
    return {"load_bytes_per_lane": counts["LDL"], "store_bytes_per_lane": counts["STL"],
            "loop_visits": visits if position is None or position >= 32 else 0,
            "position": position, "loop_branch_pc": loop.pc if loop else None,
            "method": "source iteration count x static local sites in independently partitioned SASS regions"}


def build_trace(program, launch, folder, evidence, cuobjdump):
    index = launch["index"]
    env = {**os.environ, "PATH": str(cuobjdump.parent)+os.pathsep+os.environ.get("PATH", "")}
    sass = subprocess.check_output([str(cuobjdump), "--dump-sass", str(folder/launch["module"])], text=True, env=env)
    elf = subprocess.check_output([str(cuobjdump), "--dump-elf", str(folder/launch["module"])], text=True, env=env)
    if parse(sass) != parse((folder/f"sass{index:03d}.log").read_text()):
        raise UnsupportedSpill("SASS differs from fresh cubin disassembly")
    for suffix, data in (("sass", sass), ("elf", elf)):
        with (evidence/f"{index:03d}-{suffix}.log").open("x") as f:
            f.write(data)
    q = Projection(sass)
    if not q.locals:
        raise UnsupportedSpill("No local sites to trace")
    buffers = {b["name"]: (0x100000000+(i+1)*0x100000000,
               (folder/(b["name"]+".input.bin")).read_bytes()[128:-128], b["readonly"])
               for i, b in enumerate(program["buffers"])}
    constants, abi = abi_parameters(elf, launch, buffers)
    axes = [f"SR_CTAID.{x}" in q.live_specials for x in "XYZ"]
    if set(q.live_specials) - {"SR_TID.X", "SR_CTAID.X", "SR_CTAID.Y", "SR_CTAID.Z"}:
        raise UnsupportedSpill("Unknown live special register prevents launch compression")
    multiplicity = math.prod(n for n, live in zip(launch["grid"], axes) if not live)
    if "SR_TID.X" in q.live_specials:
        # Reserved for a future demonstrated need, not a guessed SIMT schedule.
        raise UnsupportedSpill("Thread-dependent spill control/address is not yet validated")
    classes, totals, site_totals = [], Counter(), Counter()
    for cta in itertools.product(*(range(n) if live else [0] for n, live in zip(launch["grid"], axes))):
        state = Lane(q, launch, constants, buffers, cta, 0).run()
        oracle = closed_form(q, launch, buffers, cta)
        actual = {op: sum(e[3]*e[4] for e in state.events if e[1] == op) for op in ("LDL", "STL")}
        if actual != {"LDL": oracle["load_bytes_per_lane"], "STL": oracle["store_bytes_per_lane"]}:
            raise UnsupportedSpill(f"Independent local-byte count mismatch: {actual} vs {oracle}")
        if launch["entry"] == "_sink_tiled":
            check_sink_path(oracle, state.events, state.branches)
        observed = [taken for pc, taken in state.branches if pc == oracle["loop_branch_pc"]]
        if observed != [1]*max(0, oracle["loop_visits"]-1) + ([0] if oracle["loop_visits"] else []):
            raise UnsupportedSpill("Independent source/SASS loop-count mismatch")
        for pc, op, offset, count, active in state.events:
            site_totals[pc] += active*launch["threads"]*multiplicity
        for op in ("LDL", "STL"):
            totals[op] += actual[op]*launch["threads"]*multiplicity
        classes.append({"cta": list(cta), "cta_multiplicity": multiplicity,
                        "warps": launch["threads"]//32, "uniform_lanes": 32,
                        "events": state.events, "branches": state.branches, "independent_check": oracle,
                        "unobserved_padding_reads": [[pc, n] for pc, n in sorted(state.padding_reads.items())],
                        "global_offset_reads": [[name, offset, n] for (name, offset), n in sorted(state.global_reads.items())]})
    trace = {"schema": "SIM005_LOCAL_TRACE_V1", "binding": binding(program, launch),
             "engine_sha256": engine_hashes(), "entry": launch["entry"], "abi": abi,
             "sass_sha256": hashlib.sha256(sass.encode()).hexdigest(), "elf_sha256": hashlib.sha256(elf.encode()).hexdigest(),
             "cuobjdump_sha256": digest(cuobjdump), "nvdisasm_sha256": digest(cuobjdump.parent/"nvdisasm"), "projection": q.description(),
             "load_bytes": totals["LDL"], "store_bytes": totals["STL"],
             "static_sites": [{"pc": pc, "op": q.insns[pc].op, "offset": q.insns[pc].local_offset(),
                               "width": width(q.insns[pc].op), "lane_executions": site_totals[pc]} for pc in sorted(q.locals)],
             "classes": classes, "status": "COMPLETE_DYNAMIC_LOCAL_ACCESSES", "ranking_eligible": True,
             "limitations": "SASS integer/control/address projection; not numerical SASS execution, issue timing, cache measurement or H20 calibration. Uniform lanes proved from backward slice."}
    path = evidence/f"{index:03d}-local-trace.json"
    with path.open("x") as f:
        json.dump(trace, f, separators=(",", ":")); f.write("\n")
    return {"launch_index": index, "binding": trace["binding"], "trace": str(path), "trace_sha256": digest(path),
            **{k: trace[k] for k in ("status", "load_bytes", "store_bytes", "projection")}}


def load_trace(receipt, program, launch):
    path = Path(receipt["trace"])
    if digest(path) != receipt["trace_sha256"]:
        raise UnsupportedSpill("Dynamic trace hash mismatch")
    trace = json.loads(path.read_text())
    if trace["schema"] != "SIM005_LOCAL_TRACE_V1" or trace["engine_sha256"] != engine_hashes():
        raise UnsupportedSpill("Stale or unsupported dynamic trace engine")
    if trace["binding"] != binding(program, launch) or receipt["binding"] != trace["binding"]:
        raise UnsupportedSpill("Dynamic trace launch/input binding mismatch")
    if trace["status"] != "COMPLETE_DYNAMIC_LOCAL_ACCESSES" or trace["ranking_eligible"] is not True:
        raise UnsupportedSpill("Incomplete dynamic spill trace")
    sites = {s["pc"]: s for s in trace["static_sites"]}
    totals, counts, ctas = Counter(), Counter(), 0
    for group in trace["classes"]:
        if group["warps"]*32 != launch["threads"] or group["uniform_lanes"] != 32 or group["cta_multiplicity"] < 1:
            raise UnsupportedSpill("Invalid trace launch coverage")
        if launch["entry"] == "_sink_tiled":
            check_sink_path(group["independent_check"], group["events"], group["branches"])
        ctas += group["cta_multiplicity"]
        for pc, op, offset, count, active in group["events"]:
            site = sites[pc]
            if (op, offset, count) != (site["op"].split(".")[0], site["offset"], site["width"]) or active not in (0, 1):
                raise UnsupportedSpill("Local event disagrees with static site")
            n = active*group["warps"]*32*group["cta_multiplicity"]
            totals[op] += n*count; counts[pc] += n
    if ctas != math.prod(launch["grid"]) or any(s["lane_executions"] != counts[pc] for pc, s in sites.items()):
        raise UnsupportedSpill("Incomplete CTA/site coverage")
    if [totals["LDL"], totals["STL"]] != [trace["load_bytes"], trace["store_bytes"]]:
        raise UnsupportedSpill("Trace dynamic-byte summary mismatch")
    trace["trace_sha256"] = receipt["trace_sha256"]
    return trace
