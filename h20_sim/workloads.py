"""P2 launch adapters: import production kernels, keep CPU oracles independent."""

from dataclasses import dataclass, field
import sys

from pipeline import Program, Scalar
from run_swiglu import REPO, load_module, torch, triton

FOLDERS = {"gqa": "Batch-invariant_GQA", "sink": "Sink_attention",
           "moe": "FP8_MoE_forward", "router": "Router_GEMM_prefill"}


@dataclass
class Case:
    name: str
    task: str
    program: Program
    expected: dict
    tolerance: float
    metadata: dict = field(default_factory=dict)
    original_check: object = None


def call(program, fn, grid, *args, **options):
    program.launch(fn, dict(zip(fn.arg_names, args, strict=True)), grid, **options)


class Workloads:
    def __init__(self, seed):
        self.seed = seed
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.modules = {key: load_module(f"sim002_{key}_solution", REPO / folder / "solution.py")
                        for key, folder in FOLDERS.items()}
        self.oracles = {key: load_module(f"sim002_{key}_reference", REPO / folder / "reference.py")
                        for key, folder in FOLDERS.items() if key != "router"}
        # Router keeps its original independent reference/check in benchmark.py.
        # Import it without executing its __main__ block or a CUDA entry point.
        previous = sys.modules.get("solution")
        sys.path.insert(0, str(REPO / FOLDERS["router"]))
        sys.modules["solution"] = self.modules["router"]
        try:
            self.oracles["router"] = load_module("sim002_router_reference", REPO / FOLDERS["router"] / "benchmark.py")
        finally:
            sys.path.pop(0)
            if previous is None:
                sys.modules.pop("solution", None)
            else:
                sys.modules["solution"] = previous

    def random(self, shape, dtype=torch.float32, scale=0.5):
        return (torch.randn(shape, generator=self.generator) * scale).to(dtype).contiguous()

    def request(self, lq, lk, dtype, d=13, hq=4, hk=2):
        return [self.random(shape, dtype) for shape in ((lq, hq, d), (lk, hk, d), (lk, hk, d))]

    def swiglu_bf16(self, name, shape, block):
        module = load_module("sim_swiglu", REPO / "SwiGLU_backward/solution.py")
        oracle = load_module("sim_swiglu_reference", REPO / "SwiGLU_backward/reference.py")
        values = [self.random(shape, torch.bfloat16) for _ in range(3)]
        n = values[0].numel()
        production_block = 1024 if n >= 32768 else 256
        p = Program(name, "production" if block == production_block else f"diagnostic_block{block}")
        inputs = [p.buffer(key, value, readonly=True) for key, value in zip(("a", "b", "g"), values)]
        outputs = [p.output(key, shape, torch.bfloat16) for key in ("da", "db")]
        call(p, module._backward, (triton.cdiv(n, block),), *inputs, *outputs, n, block, num_warps=4)
        return Case(name, "swiglu", p, dict(zip(("da", "db"), oracle.reference(*values))), 1e-2,
                    {"shape": list(shape), "dtype": str(torch.bfloat16)})

    def gqa(self, name, requests, mode="full", window=17, invariant=None):
        q, k, v = [torch.cat([request[i] for request in requests]) for i in range(3)]
        qo, ko = [0], [0]
        for x, y, _ in requests:
            qo.append(qo[-1] + x.shape[0]); ko.append(ko[-1] + y.shape[0])
        values = [q, k, v, torch.tensor(qo, dtype=torch.int32), torch.tensor(ko, dtype=torch.int32)]
        p = Program(name)
        refs = [p.buffer(n, x, readonly=True) for n, x in zip(("q", "k", "v", "qoff", "koff"), values)]
        out = p.output("output", q.shape, q.dtype)
        nq, hq, d = q.shape
        hk = k.shape[1]
        bn = 32 if mode == "sliding" and window <= 32 else (128 if d <= 128 else 64)
        call(p, self.modules["gqa"]._gqa, (nq, hq), *refs, out, Scalar(len(requests)),
             hq, hk, d, window if mode == "sliding" else 0, triton.next_power_of_2(d), bn, num_warps=4)
        expected = self.oracles["gqa"].reference(*values, mode=mode, window=window)
        return Case(name, "gqa", p, {"output": expected}, 5e-3 if q.dtype == torch.bfloat16 else 2e-5,
                    {"mode": mode, "window": window, "q_offsets": qo, "kv_offsets": ko,
                     "invariance": invariant, "dtype": str(q.dtype), "shape": list(q.shape)})

    def sink(self, name, dtype, *, tiled=False, mode="full", shape=None):
        if shape is None:
            shape = (1, 4, 17, 4, 1, 128) if tiled else (2, 3, 33, 4, 2, 13)
        b, lq, lk, hq, hk, d = shape
        values = [self.random(shape, dtype) for shape in
                  ((b, lq, hq, d), (b, lk, hk, d), (b, lk, hk, d), (hk, d))]
        p = Program(name)
        q, k, v, s = [p.buffer(n, x, readonly=True) for n, x in zip(("q", "k", "v", "sink"), values)]
        out = p.output("output", values[0].shape, dtype)
        window = 17
        if dtype == torch.bfloat16 and d == 128 and lq > 1 and lq * (hq // hk) >= 16:
            call(p, self.modules["sink"]._sink_tiled, (triton.cdiv(lq * (hq // hk), 16), hk, b),
                 q, k, v, s, out, lq, lk, hq, hk, d, window if mode == "sliding" else 0,
                 128, 16, 32, 3, num_warps=4, num_stages=2)
        else:
            bn = 32 if lk <= 32 else (128 if d <= 128 else 64)
            call(p, self.modules["sink"]._sink, (b * lq, hq), q, k, v, s, out,
                 lq, lk, hq, hk, d, window if mode == "sliding" else 0,
                 triton.next_power_of_2(d), bn, num_warps=4)
        expected = self.oracles["sink"].reference(*values, mode=mode, window=window)
        return Case(name, "sink", p, {"output": expected}, 5e-3 if dtype == torch.bfloat16 else 2e-5,
                    {"mode": mode, "window": window, "shape": list(values[0].shape), "dtype": str(dtype)})

    def moe(self, name, dtype, *, t=3, e=4, d=33, width=31, selected=3, grouped=False):
        x = self.random((t, d), dtype, 0.25)
        wgu = self.random((e, 2 * width, d), torch.float8_e4m3fn, 0.25)
        wd = self.random((e, d, width), torch.float8_e4m3fn, 0.25)
        ids = ((torch.arange(t)[:, None] + torch.arange(selected)[None, :]) % e).to(torch.int32)
        weights = torch.rand((t, selected), generator=self.generator)
        weights = weights / weights.sum(-1, keepdim=True)
        sg = torch.linspace(0.25, 0.75, e)
        sd = torch.linspace(0.75, 0.25, e)
        values = [x, wgu, wd, ids, weights, sg, sd]
        production_grouped = dtype == torch.bfloat16 and selected == 8 and t >= max(32, e) and d >= 1024 and width >= 512
        p = Program(name, "production" if not grouped or production_grouped else "diagnostic_grouped_smaller_shape")
        X, WG, WD, IDS, P, SG, SD = [p.buffer(n, z, readonly=True) for n, z in
                                    zip(("x", "wgu", "wdown", "ids", "weights", "scale_gu", "scale_down"), values)]
        out = p.output("output", (t, d), dtype)
        if t:
            h = p.output("hidden", (t, selected, width), torch.float32)
            m = self.modules["moe"]
            if grouped or production_grouped:
                partial = out if selected == 1 else p.output("partial", (t, selected, d), torch.float32)
                count = p.output("count", (e,), torch.int32)
                routes = p.output("routes", (e, t), torch.int32)
                call(p, m._bucket_reset, (triton.cdiv(e, 256),), count, e, 256, num_warps=4)
                call(p, m._bucket_routes, (triton.cdiv(t * selected, 256),), IDS, count, routes, t, selected, 256, num_warps=4)
                call(p, m._grouped_gate_up, (e, triton.cdiv(t, 16), triton.cdiv(width, 64)),
                     X, WG, SG, count, routes, h, t, d, width, selected, 16, 64, 64, num_warps=4)
                call(p, m._grouped_down, (e, triton.cdiv(t, 16), triton.cdiv(d, 64)),
                     h, WD, SD, count, routes, partial, P, t, d, width, 16, 64, 64, 3, selected == 1, num_warps=4)
                if selected != 1:
                    call(p, m._combine, (t, triton.cdiv(d, 128)), partial, P, out, d, selected,
                         128, triton.next_power_of_2(selected), num_warps=4)
            else:
                fused = 1 < selected <= 8 and d <= 128 and width <= 64
                partial = out if selected == 1 or fused else p.output("partial", (t, selected, d), torch.float32)
                call(p, m._gate_up, (t * selected, triton.cdiv(width, 32 if width >= 128 else 16)),
                     X, WG, IDS, SG, h, d, width, selected, 32 if width >= 128 else 16,
                     min(256, triton.next_power_of_2(d)), num_warps=4)
                if fused:
                    call(p, m._down_combine, (t, triton.cdiv(d, 8)), h, WD, IDS, SD, P, out,
                         d, width, selected, 8, min(256, triton.next_power_of_2(width)),
                         triton.next_power_of_2(selected), num_warps=4)
                else:
                    call(p, m._down, (t * selected, triton.cdiv(d, 32 if d >= 256 else 16)),
                         h, WD, IDS, SD, partial, P, d, width, 32 if d >= 256 else 16,
                         min(256, triton.next_power_of_2(width)), selected == 1, num_warps=4)
                    if selected != 1:
                        call(p, m._combine, (t, triton.cdiv(d, 128)), partial, P, out, d, selected,
                             128, triton.next_power_of_2(selected), num_warps=4)
        expected = self.oracles["moe"].reference(*values)
        independent = self.oracles["moe"].grouped_reference(*values)
        if not torch.equal(expected, independent):
            from common.validation import compare
            compare(expected, independent, atol=2e-2 if dtype == torch.bfloat16 else 1e-4,
                    rtol=2e-2 if dtype == torch.bfloat16 else 1e-4, name="moe_oracles")
        return Case(name, "moe", p, {"output": expected}, 2e-2 if dtype == torch.bfloat16 else 1e-4,
                    {"shape": [t, e, d, width, selected], "dtype": str(dtype),
                     "bucket_check": grouped or production_grouped, "independent_oracle_passed": True})

    def router(self, name, tokens, *, packed_diagnostic=False):
        x, w = self.random((tokens, 3072), torch.bfloat16), self.random((256, 3072))
        p = Program(name, "diagnostic_packed_smaller_shape" if packed_diagnostic and tokens < 2048 else "production")
        X, W = p.buffer("x", x, readonly=True), p.buffer("w", w, readonly=True)
        out = p.output("output", (tokens, 256), torch.float32)
        m = self.modules["router"]
        if 0 < tokens <= 4 and not packed_diagnostic:
            call(p, m._gemv, (tokens, 128), X, W, out, 2, 4096, num_warps=4)
        elif tokens >= 2048 or packed_diagnostic:
            h = p.output("high", w.shape, torch.bfloat16)
            l = p.output("low", w.shape, torch.float16)
            call(p, m._pack, (triton.cdiv(256 * 3072, 1024),), W, h, l, 1024, num_warps=4)
            call(p, m._packed_gemm, (triton.cdiv(tokens, 64) * 4, 1), X, h, l, out,
                 tokens, 64, 64, 64, 1, True, False, num_warps=4, num_stages=3)
        elif tokens:
            bm, bn, bk, split = ((16, 32, 64, 8) if tokens <= 16 else
                                 (32, 64, 32, 8) if tokens <= 128 else
                                 (32, 64, 32, 4) if tokens <= 1024 else (32, 64, 32, 1))
            partial = out if split == 1 else p.output("partial", (split, tokens, 256), torch.float32)
            call(p, m._gemm, (triton.cdiv(tokens, bm) * (256 // bn), split),
                 X, W, partial, tokens, bm, bn, bk, split, tokens >= 1024, num_warps=4, num_stages=3)
            if split != 1:
                call(p, m._reduce, (triton.cdiv(tokens * 256, 256),), partial, out, tokens, split, 256, num_warps=4)
        oracle = self.oracles["router"]
        return Case(name, "router", p, {"output": oracle.reference(x, w)}, 2e-3,
                    {"tokens": tokens, "dtype": "torch.bfloat16", "weight_shape": [256, 3072]}, oracle.check)

    def cases(self, suite):
        if suite == "smoke":
            yield self.gqa("gqa_bf16", [self.request(2, 17, torch.bfloat16)])
            yield self.sink("sink_scalar_bf16", torch.bfloat16)
            yield self.sink("sink_tiled_bf16", torch.bfloat16, tiled=True)
            yield self.moe("moe_fused_bf16", torch.bfloat16)
            yield self.router("router_tiny", 1)
            return
        if suite == "extended":
            yield self.gqa("gqa_d128_production", [self.request(1, 33, torch.bfloat16, d=128)])
            yield self.moe("moe_grouped_production", torch.bfloat16, t=32, e=8, d=1024, width=512, selected=8)
            yield self.router("router_packed_diagnostic", 2, packed_diagnostic=True)
            return
        for dtype in (torch.bfloat16, torch.float32):
            tag = str(dtype).split(".")[-1]
            for mode in ("full", "sliding"):
                target = self.request(3, 33, dtype)
                neighbor = self.request(2, 7, dtype)
                group = f"{self.seed}-{tag}-{mode}"
                yield self.gqa(f"gqa_{tag}_{mode}_alone", [target], mode,
                               invariant={"group": group, "role": "base", "output_slice": [0, 3], "base_slice": [0, 3]})
                yield self.gqa(f"gqa_{tag}_{mode}_batch", [neighbor, target], mode,
                               invariant={"group": group, "role": "compare", "output_slice": [2, 5], "base_slice": [0, 3]})
                for a, b in ((0, 1), (1, 3)):
                    request = [target[0][a:b].contiguous(), target[1][:30+b].contiguous(), target[2][:30+b].contiguous()]
                    yield self.gqa(f"gqa_{tag}_{mode}_chunk_{a}_{b}", [request], mode,
                                   invariant={"group": group, "role": "compare", "output_slice": [0, b-a], "base_slice": [a, b]})
                yield self.sink(f"sink_{tag}_{mode}", dtype, mode=mode)
            yield self.moe(f"moe_{tag}_top1", dtype, selected=1, d=17, width=15)
            yield self.moe(f"moe_{tag}_fused", dtype)
            yield self.moe(f"moe_{tag}_three_launches", dtype, t=2, d=129, width=65, selected=2)
            yield self.moe(f"moe_{tag}_empty", dtype, t=0)
        for mode in ("full", "sliding"):
            yield self.sink(f"sink_tiled_{mode}", torch.bfloat16, tiled=True, mode=mode)
        yield self.moe("moe_grouped_diagnostic", torch.bfloat16, t=5, e=4, d=32, width=16, selected=3, grouped=True)
        yield self.router("router_tiny", 1)
        yield self.router("router_split_reduce", 5)
