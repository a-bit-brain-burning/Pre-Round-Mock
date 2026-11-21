"""Exact FMA, persistent alias and predicated atomic CPU backend regressions."""

import argparse
import json
import os
from pathlib import Path

from bootstrap import DEFAULT_ROOT, HERE, digest, prepare, timestamp, validated_build
from pipeline import Program, Ref, compile_harness, execute, read_snapshot
from operator_source import source_key
from run_swiglu import REPO, runtime_environment, torch, triton, write_json
from workloads import Workloads, call
from run_suite import bucket_check
import triton.language as tl


@triton.jit
def _fma_probe(A, B, C, FUSED, SEPARATE, N: tl.constexpr, WIDE: tl.constexpr):
    i = tl.arange(0, 128)
    a, b, c = tl.load(A + i, i < N, 0), tl.load(B + i, i < N, 0), tl.load(C + i, i < N, 0)
    if WIDE:
        fused = tl.inline_asm_elementwise("fma.rn.f64 $0, $1, $2, $3;", "=d,d,d,d",
                                          [a, b, c], tl.float64, is_pure=True, pack=1)
        separate = tl.inline_asm_elementwise(
            "{.reg .f64 t; mul.rn.f64 t,$1,$2; add.rn.f64 $0,t,$3;}", "=d,d,d,d",
            [a, b, c], tl.float64, is_pure=True, pack=1)
    else:
        fused = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                                          [a, b, c], tl.float32, is_pure=True, pack=1)
        separate = tl.inline_asm_elementwise(
            "{.reg .f32 t; mul.rn.f32 t,$1,$2; add.rn.f32 $0,t,$3;}", "=f,f,f,f",
            [a, b, c], tl.float32, is_pure=True, pack=1)
    tl.store(FUSED + i, fused, i < N)
    tl.store(SEPARATE + i, separate, i < N)


@triton.jit
def _produce(X, T, N: tl.constexpr):
    i = tl.arange(0, 128)
    x = tl.load(X + i, i < N, 0)
    tl.store(T + i, 2 * x + 1, i < N)


@triton.jit
def _consume(T, Y, N: tl.constexpr):
    i = tl.arange(0, 128)
    t = tl.load(T + i, i < N, 0)
    tl.store(Y + i, 3 * t - 4, i < N)


def fma_program(dtype):
    wide = dtype == torch.float64
    epsilon = 2.0 ** (-27 if wide else -23)
    tiny = 2.0 ** (-1074 if wide else -149)
    maximum = torch.finfo(dtype).max
    p = Program(f"fma_{dtype}", dispatch="instruction_probe")
    data = ([1 + epsilon, 1 + epsilon, maximum, tiny, 1, -1],
            [1 - epsilon, 1 + epsilon, 2, 1, 1, 1],
            [-1, -(1 + 2 * epsilon), -maximum, 0, -1, 1])
    refs = [p.buffer(n, torch.tensor(v, dtype=dtype), readonly=True) for n, v in zip(("a", "b", "c"), data)]
    fused, separate = p.output("fused", (6,), dtype), p.output("separate", (6,), dtype)
    p.launch(_fma_probe, dict(A=refs[0], B=refs[1], C=refs[2], FUSED=fused, SEPARATE=separate, N=6, WIDE=wide), (1,))
    expected = torch.tensor([-epsilon * epsilon, epsilon * epsilon, maximum, tiny, 0, 0], dtype=dtype)
    return p, expected


def dependency_program(omit_producer=False):
    p = Program("missing_producer" if omit_producer else "persistent_alias", dispatch="dependency_probe")
    x = p.buffer("input", torch.arange(5, dtype=torch.float32), readonly=True)
    temp = p.output("temp", (10,), torch.float32)
    y = p.output("output", (5,), torch.float32)
    view = Ref(temp.name, 5 * 4)
    if not omit_producer:
        p.launch(_produce, dict(X=x, T=view, N=5), (1,))
    p.launch(_consume, dict(T=view, Y=y, N=5), (1,))
    return p


def bucket_program(module, size, pattern):
    p = Program(f"atomic_{size}_{pattern}", dispatch="production_bucket_kernel_diagnostic_shape")
    values = torch.arange(size, dtype=torch.int32) % 4 if pattern == "striped" else torch.zeros(size, dtype=torch.int32)
    ids = p.buffer("ids", values.reshape(size, 1), readonly=True)
    counts = p.output("count", (4,), torch.int32)
    routes = p.output("routes", (4, size), torch.int32)
    call(p, module._bucket_reset, (1,), counts, 4, 256, num_warps=4)
    # The empty diagnostic intentionally launches one fully masked block.
    call(p, module._bucket_routes, (max(1, triton.cdiv(size, 256)),),
         ids, counts, routes, size, 1, 256, num_warps=4)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--label", default="audit")
    args = ap.parse_args()
    torch.set_num_threads(1)
    root = prepare(args.root)
    run = root / "runs" / f"sim002-probes-{timestamp()}"
    run.mkdir()
    os.environ["TRITON_CACHE_DIR"] = str(run / "triton-cache")
    source, library, env = runtime_environment(root)
    exe = compile_harness(run, library, env)
    rows = []
    for dtype in (torch.float32, torch.float64):
        p, expected = fma_program(dtype)
        folder = run / str(dtype).split(".")[-1]
        folder.mkdir()
        evidence, outputs = execute(p, folder, source, library, env, exe)
        bits = torch.int32 if dtype == torch.float32 else torch.int64
        passed = torch.equal(outputs["fused"].view(bits), expected.view(bits))
        row = {"probe": p.label, "passed": passed, "expected": expected.tolist(),
               "actual": outputs["fused"].tolist(), "separate": outputs["separate"].tolist(),
               "artifacts": str(folder), **evidence}
        rows.append(row)
        print(json.dumps({"probe": p.label, "passed": passed}), flush=True)
    for omit in (False, True):
        p = dependency_program(omit)
        folder = run / p.label
        folder.mkdir()
        evidence, outputs = execute(p, folder, source, library, env, exe)
        expected = 6 * torch.arange(5, dtype=torch.float32) - 1
        if omit:
            passed = bool(torch.isnan(outputs["output"]).all()) and not torch.equal(outputs["output"], expected)
        else:
            early = read_snapshot(folder / "step000-output.bin", p.buffers["output"]["tensor"])
            passed = (torch.equal(outputs["output"], expected) and bool(torch.isnan(early).all())
                      and bool(torch.isnan(outputs["temp"][:5]).all()))
        rows.append({"probe": p.label, "passed": passed, "artifacts": str(folder), **evidence})
        print(json.dumps({"probe": p.label, "passed": passed}), flush=True)
    module = Workloads(1108).modules["moe"]
    for size in (0, 1, 15, 31, 32, 33, 127, 128, 129, 255, 256, 257):
        for pattern in ("contended", "striped"):
            p = bucket_program(module, size, pattern)
            folder = run / p.label
            folder.mkdir()
            evidence, outputs = execute(p, folder, source, library, env, exe)
            checks = bucket_check(p, outputs)
            reset = read_snapshot(folder / "step000-count.bin", p.buffers["count"]["tensor"])
            passed = checks["passed"] and bool((reset == 0).all())
            rows.append({"probe": p.label, "passed": passed, "bucket_checks": checks,
                         "artifacts": str(folder), **evidence})
            print(json.dumps({"probe": p.label, "passed": passed}), flush=True)
    report = {"experiment": "sim002", "kind": "instruction_and_dependency_probes", "label": args.label,
              "status": "PASS" if all(r["passed"] for r in rows) else "HAS_FAILURES", "cases": rows,
              "build": validated_build(root, source, library), "gpu_execution": False, "h20_tested": False,
              "source_sha256": {source_key(p): digest(p) for p in
                                (Path(__file__), HERE / "pipeline.py", HERE / "pipeline.cpp",
                                 HERE / "workloads.py", HERE / "run_suite.py", HERE / "patch_backend.py",
                                 REPO / "FP8_MoE_forward/solution.py")}}
    output = HERE / f"results-sim002-probes-{timestamp()[:15]}.json"
    write_json(output, report)
    print(f"{report['status']}; {output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
