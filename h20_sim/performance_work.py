"""Source-bound work accounting for the five contest kernels only.

Logical requests are not cache transactions. Dot FLOPs count FMA as two;
auxiliary arithmetic, conversions, indexing and shared traffic are proxies.
The model driver verifies source hashes before calling this module.
"""

import math

TASKS = {"swiglu", "gqa", "sink", "moe", "router"}
WIDTH = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4,
         "torch.float64": 8, "torch.int32": 4, "torch.int64": 8,
         "torch.float8_e4m3fn": 1, "torch.uint8": 1}


def ceildiv(n, d):
    return (n + d - 1) // d


class Work:
    def __init__(self, launch, buffers):
        self.launch = launch
        self.buffers = buffers
        self.ctas = math.prod(launch["grid"])
        self.active = self.ctas
        self.ops = dict.fromkeys(("fp32", "fp64", "tensor", "exp2", "div32", "div64",
                                 "convert", "integer", "atomic"), 0)
        self.traffic = {}
        self.max_loop_visits = 1
        self.mean_loop_visits = 1
        self.max_atomic_chain = 0
        self.reductions = 0
        self.notes = []

    def read(self, name, elements, unique=None):
        b = self.buffers[name]
        width = WIDTH[b["dtype"]]
        unique = min(elements, math.prod(b["shape"])) if unique is None else unique
        if not 0 <= unique <= elements or unique > math.prod(b["shape"]):
            raise ValueError(f"Invalid logical unique reads: {name}")
        if name in self.traffic:
            raise ValueError(f"Account each buffer once: {name}")
        self.traffic[name] = {"read_bytes": elements * width, "unique_read_bytes": unique * width,
                              "write_bytes": 0, "intermediate": not b["readonly"]}

    def write(self, name, elements):
        if not 0 <= elements <= math.prod(self.buffers[name]["shape"]):
            raise ValueError(f"Invalid logical stores: {name}")
        v = self.traffic.setdefault(name, {"read_bytes": 0, "unique_read_bytes": 0,
                                           "write_bytes": 0, "intermediate": True})
        v["write_bytes"] = elements * WIDTH[self.buffers[name]["dtype"]]

    def finish(self):
        # Address/control count is a source-level proxy; it is intentionally
        # separated from exact valid-element traffic and contraction arithmetic.
        if not self.ops["integer"]:
            self.ops["integer"] = 4 * sum(
                (v["read_bytes"] + v["write_bytes"]) / WIDTH[self.buffers[n]["dtype"]]
                for n, v in self.traffic.items())
        if not 0 <= self.active <= self.ctas or self.max_loop_visits < 1:
            raise ValueError("Invalid schedule")
        if any(not math.isfinite(v) or v < 0 for v in self.ops.values()):
            raise ValueError("Invalid operation count")
        shared = self.launch["shared_bytes"] + self.launch["resources"].get("shared", 0)
        return {"entry": self.launch["entry"], "launched_ctas": self.ctas, "active_ctas": self.active,
                "empty_ctas": self.ctas - self.active, "operations": self.ops,
                "traffic": self.traffic, "logical_bytes": sum(v["read_bytes"] + v["write_bytes"] for v in self.traffic.values()),
                "max_loop_visits": self.max_loop_visits, "mean_loop_visits": self.mean_loop_visits,
                "max_atomic_chain": self.max_atomic_chain, "reduction_depth_proxy": self.reductions,
                "shared_bytes_proxy": 2 * shared * self.active * (self.mean_loop_visits + 1),
                "notes": self.notes,
                "counting_basis": "source logical valid elements and padded arithmetic; auxiliary/shared counts are proxies"}


def attention(w, c, task, aux):
    b = w.buffers
    hq, hk, d, bd, bn, win = (c[k] for k in ("HQ", "HK", "D", "BD", "BN", "WINDOW"))
    if b["q"]["shape"][-2:] != [hq, d] or b["k"]["shape"][-2:] != [hk, d] or b["v"]["shape"] != b["k"]["shape"]:
        raise ValueError("Attention constants do not match buffers")
    wide = task == "gqa" and b["q"]["dtype"] == "torch.bfloat16"
    if task == "gqa":
        qo, ko = aux["q_offsets"], aux["kv_offsets"]
        if qo[0] or ko[0] or qo[-1] != b["q"]["shape"][0] or ko[-1] != b["k"]["shape"][0]:
            raise ValueError("Offsets do not match input tensors")
        lengths = [(qe - qs, ke - ks) for qs, qe, ks, ke in zip(qo, qo[1:], ko, ko[1:])]
        if any(not 1 <= lq <= lk for lq, lk in lengths):
            raise ValueError("Invalid attention lengths")
        batches = 1
    else:
        batches, lq, _, _ = b["q"]["shape"]
        if lq != c["LQ"] or b["k"]["shape"][:2] != [batches, c["LK"]]:
            raise ValueError("Sink lengths do not match buffers")
        lengths = [(lq, c["LK"])] * batches
        qo = [0]
        for lq, _ in lengths:
            qo.append(qo[-1] + lq)
    nq = sum(lq for lq, _ in lengths)
    w.read("q", nq * hq * d)
    w.write("output", nq * hq * d)
    loops, slots, pairs, unique_keys, search_reads = 0, 0, 0, 0, 0
    maximum = 1
    if w.launch["entry"] == "_sink_tiled":
        bm, parts = c["BM"], c["P_PARTS"]
        if parts != 3:
            raise ValueError("Only the current three-part Sink path is covered")
        lq, lk = lengths[0]
        group = hq // hk
        if w.launch["grid"] != [ceildiv(lq * group, bm), hk, batches]:
            raise ValueError("Sink tiled grid mismatch")
        loaded = 0
        for tile in range(ceildiv(lq * group, bm)):
            first = tile * bm // group
            last = min(lq - 1, (tile * bm + bm - 1) // group)
            left = max(0, lk - lq + first - win + 1) if win else 0
            start = left // bn * bn
            visits = ceildiv(lk - lq + last + 1 - start, bn)
            loops += visits * batches * hk
            loaded += (min(lk, start + visits * bn) - start) * batches * hk
            maximum = max(maximum, visits)
        start = (max(0, lk - lq - win + 1) if win else 0) // bn * bn
        unique_keys = batches * hk * (lk - start)
        w.read("k", loaded * d, unique_keys * d)
        w.read("v", loaded * d, unique_keys * d)
        w.read("sink", w.ctas * d)
        slots = loops * bm * bn
        w.ops["tensor"] = 2 * (1 + parts) * slots * bd
        w.ops["fp32"] = slots * 7 + loops * bm * (bd + 4) + w.ctas * bm * 2 * bd
        w.ops["exp2"] = loops * bm * (bn + 1)
        w.ops["div32"] = w.ctas * bm * bd
        w.ops["convert"] = 5 * slots
        w.reductions = 2 * math.ceil(math.log2(bn))
    else:
        if w.ctas != nq * hq:
            raise ValueError("Attention grid does not match source")
        for request, (lq, lk) in enumerate(lengths):
            unique_keys += hk * (min(lk, lq + win - 1) if win else lk)
            for q in range(lq):
                position = lk - lq + q
                local_bn = 32 if task == "gqa" and position < 32 else bn
                left = max(0, position - win + 1) if win else 0
                visits = position // local_bn - left // local_bn + 1
                loops += visits * hq
                slots += visits * local_bn * hq
                pairs += (position - left + 1) * hq
                maximum = max(maximum, visits)
                w.ops["fp32"] += hq * visits * (2 * local_bn * bd + 2 * local_bn + 1)
                w.ops["fp64" if wide else "fp32"] += hq * visits * (2 * local_bn * bd + bd + local_bn + 1)
                w.ops["exp2"] += hq * visits * (local_bn + 1)
                if task == "gqa":
                    lo, hi = 0, len(lengths)
                    while lo + 1 < hi:
                        mid = (lo + hi) // 2
                        search_reads += hq
                        if qo[request] + q >= qo[mid]:
                            lo = mid
                        else:
                            hi = mid
        w.read("k", pairs * d, unique_keys * d)
        w.read("v", pairs * d, unique_keys * d)
        w.ops["div64" if wide else "div32"] = nq * hq * bd
        if task == "gqa":
            w.read("qoff", search_reads + 2 * nq * hq)
            w.read("koff", 2 * nq * hq)
            maximum += math.ceil(math.log2(len(lengths)))
        else:
            w.read("sink", nq * hq * d)
            w.ops["fp32"] += 3 * nq * hq * bd
        w.ops["convert"] = (3 * slots * bd if wide else 2 * slots * bd) if b["q"]["dtype"] == "torch.bfloat16" else 0
        w.reductions = math.ceil(math.log2(bd)) + 2 * math.ceil(math.log2(bn))
    w.max_loop_visits = maximum
    w.mean_loop_visits = loops / w.ctas
    w.notes.append("Masks reduce valid memory traffic; arithmetic includes scheduled/padded lanes. FP64 state and divisions are separate from FP32 and Tensor compute.")


def router(w, c):
    name, b = w.launch["entry"], w.buffers
    m = b["x"]["shape"][0]
    size = 256 * 3072
    if name == "_pack":
        if w.ctas != ceildiv(size, c["BLOCK"]):
            raise ValueError("Router pack grid mismatch")
        w.read("w", size)
        w.write("high", size)
        w.write("low", size)
        w.ops.update(fp32=2 * size, convert=3 * size)
    elif name == "_gemv":
        bn, bk = c["BN"], c["BK"]
        if 256 % bn or w.launch["grid"][:2] != [m, 256 // bn]:
            raise ValueError("Router GEMV grid mismatch")
        w.read("x", m * 3072 * (256 // bn))
        w.read("w", m * size)
        w.write("output", m * 256)
        w.ops["fp32"] = 2 * w.ctas * bn * ceildiv(3072, bk) * bk + w.ctas * bn * (bk - 1)
        w.ops["convert"] = m * 3072 * (256 // bn)
        w.max_loop_visits = w.mean_loop_visits = ceildiv(3072, bk)
        w.reductions = math.ceil(math.log2(bk))
    elif name in ("_gemm", "_packed_gemm"):
        bm, bn, bk, split = (c[k] for k in ("BM", "BN", "BK", "SPLIT"))
        if c.get("MIXED") or 256 % bn or 3072 % (split * bk) or c["M"] != m:
            raise ValueError("Unsupported Router launch")
        visits = 3072 // (split * bk)
        expected_ctas = ceildiv(m, bm) * (256 // bn) * split
        if w.ctas != expected_ctas:
            raise ValueError("Router grid mismatch")
        w.read("x", m * 3072 * (256 // bn))
        tiles = ceildiv(m, bm)
        if name == "_packed_gemm":
            w.read("high", size * tiles)
            w.read("low", size * tiles)
        else:
            w.read("w", size * tiles)
        w.write("partial" if split > 1 else "output", split * m * 256)
        w.ops["tensor"] = 4 * w.ctas * bm * bn * bk * visits
        w.ops["fp32"] = w.ctas * bm * bn * (visits + 2 if c["LOW_ACC"] else 1)
        if name == "_gemm":
            w.ops["fp32"] += size * tiles * (2 if c["LOW_ACC"] else 1)
        w.ops["convert"] = (m * 3072 * (256 // bn) + w.ctas * bm * bn * visits) if c["LOW_ACC"] else 0
        if name == "_gemm":
            w.ops["convert"] += 3 * size * tiles
        w.max_loop_visits = w.mean_loop_visits = visits
        w.notes.append("Two dense Tensor contractions; FP16 accumulator throughput shares the BF16 capacity assumption. Pack runs on every call.")
    elif name == "_reduce":
        split, block = c["SPLIT"], c["BLOCK"]
        if c["M"] != m or w.ctas != ceildiv(m * 256, block):
            raise ValueError("Router reduce grid mismatch")
        w.read("partial", split * m * 256)
        w.write("output", m * 256)
        w.ops["fp32"] = w.ctas * block * (split - 1)
        w.reductions = math.ceil(math.log2(split))
    else:
        raise ValueError(f"Unmodeled Router kernel: {name}")


def moe(w, c, aux):
    b, name = w.buffers, w.launch["entry"]
    t, d = b["x"]["shape"]
    e, _, width = b["wdown"]["shape"]
    k = b["ids"]["shape"][1]
    r = t * k
    if any(c[key] != value for key, value in {"T": t, "D": d, "I": width, "K": k, "E": e}.items() if key in c):
        raise ValueError("MoE constants do not match buffers")
    counts = aux["expert_counts"]
    if len(counts) != e or sum(counts) != r or any(n < 0 or n > t for n in counts):
        raise ValueError("Invalid distinct-expert route histogram")
    experts = sum(n > 0 for n in counts)
    if name == "_bucket_reset":
        w.write("count", e)
    elif name == "_bucket_routes":
        w.read("ids", r)
        w.read("count", r, experts)
        # Atomics can store to the same address repeatedly; account explicitly.
        w.traffic["count"]["write_bytes"] = 4 * r
        w.write("routes", r)
        w.ops["atomic"] = r
        w.max_atomic_chain = max(counts, default=0)
    elif name == "_combine":
        w.read("partial", r * d)
        w.read("weights", r * ceildiv(d, c["BOUT"]))
        w.write("output", t * d)
        w.ops["fp32"] = w.ctas * c["BOUT"] * (2 * c["BSEL"] - 1)
        w.reductions = math.ceil(math.log2(c["BSEL"]))
    elif name in ("_grouped_gate_up", "_grouped_down"):
        bm, bn, bk = c["BM"], c["BN"], c["BK"]
        gate = name == "_grouped_gate_up"
        inner, outer = (d, width) if gate else (width, d)
        active_m = sum(ceildiv(n, bm) for n in counts)
        ntiles = ceildiv(outer, bn)
        if w.launch["grid"] != [e, ceildiv(t, bm), ntiles]:
            raise ValueError("Grouped MoE grid mismatch")
        w.active = active_m * ntiles
        w.read("count", w.ctas)
        w.read("routes", r * ntiles, r)
        w.read("x" if gate else "hidden", r * inner * ntiles)
        w.read("wgu" if gate else "wdown", (2 if gate else 1) * active_m * d * width,
               (2 if gate else 1) * experts * d * width)
        w.read("scale_gu" if gate else "scale_down", w.active, experts)
        w.write("hidden" if gate else ("output" if c["FUSE_SINGLE"] else "partial"), r * outer)
        if not gate and c["FUSE_SINGLE"]:
            w.read("weights", r * ntiles)
        terms = 2 if gate else c["TERMS"]
        if not gate and terms != 3:
            raise ValueError("Only validated three-term grouped MoE is covered")
        visits = ceildiv(inner, bk)
        padded = w.active * bm * bn
        w.ops["tensor"] = 2 * terms * padded * visits * bk
        w.ops["fp32"] = (5 if gate else 3) * padded + (0 if gate else 2 * w.active * bm * visits * bk)
        w.ops["convert"] = (2 if gate else 1) * w.active * bn * visits * bk + (0 if gate else 5 * w.active * bm * visits * bk)
        if gate:
            w.ops["exp2"] = w.ops["div32"] = padded
        w.max_loop_visits = w.mean_loop_visits = visits
        w.notes.append("Inactive expert tiles still load COUNT. FP8 weights are converted to BF16; the Tensor rate is BF16, not native FP8.")
    elif name in ("_gate_up", "_down", "_down_combine"):
        gate = name == "_gate_up"
        fused = name == "_down_combine"
        inner, outer = (d, width) if gate else (width, d)
        bo, bk = c["BO"], c["BK"]
        nt = ceildiv(outer, bo)
        if w.launch["grid"][:2] != [t if fused else r, nt]:
            raise ValueError("Slot MoE grid mismatch")
        w.read("ids", r * nt)
        w.read("scale_gu" if gate else "scale_down", r * nt, experts)
        w.read("x" if gate else "hidden", r * inner * nt)
        w.read("wgu" if gate else "wdown", (2 if gate else 1) * r * d * width,
               (2 if gate else 1) * experts * d * width)
        if not gate and (fused or c["FUSE_SINGLE"]):
            w.read("weights", r * nt)
        w.write("hidden" if gate else ("output" if fused or c["FUSE_SINGLE"] else "partial"),
                (t if fused else r) * outer)
        padded = w.ctas * bo * (c["BSEL"] if fused else 1)
        visits = ceildiv(inner, bk)
        w.ops["fp32"] = (4 if gate else 2) * padded * visits * bk + (5 if gate else 2) * padded
        w.ops["convert"] = (2 if gate else 1) * padded * visits * bk
        if gate:
            w.ops["exp2"] = w.ops["div32"] = padded
        w.max_loop_visits = w.mean_loop_visits = visits
        w.reductions = math.ceil(math.log2(bk)) + (math.ceil(math.log2(c["BSEL"])) if fused else 0)
    else:
        raise ValueError(f"Unmodeled MoE kernel: {name}")


def account(task, program, aux):
    if task not in TASKS:
        raise ValueError("Simulator scope is exactly the five preliminary-contest tasks")
    buffers = {b["name"]: b for b in program["buffers"]}
    rows = []
    for launch in program["launches"]:
        w, c = Work(launch, buffers), launch["constants"]
        if task == "swiglu":
            if launch["entry"] != "_backward":
                raise ValueError("Unknown SwiGLU kernel")
            n, block = c["N"], c["BLOCK"]
            if n != math.prod(buffers["a"]["shape"]):
                raise ValueError("SwiGLU size differs from input")
            if w.ctas != ceildiv(n, block):
                raise ValueError("SwiGLU grid mismatch")
            for name in ("a", "b", "g"):
                w.read(name, n)
            for name in ("da", "db"):
                w.write(name, n)
            padded = w.ctas * block
            w.ops.update(fp32=11 * padded, exp2=padded, div32=padded,
                         convert=5 * padded if buffers["a"]["dtype"] == "torch.bfloat16" else 0)
        elif task in ("gqa", "sink"):
            if launch["entry"] not in ({"_gqa"} if task == "gqa" else {"_sink", "_sink_tiled"}):
                raise ValueError("Unknown attention kernel")
            attention(w, c, task, aux)
        elif task == "router":
            router(w, c)
        else:
            moe(w, c, aux)
        rows.append(w.finish())
    return rows
