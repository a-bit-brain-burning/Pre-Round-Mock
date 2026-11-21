"""CPU validation of the exact WGMMA variants needed by Router packed."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from bootstrap import DEFAULT_ROOT, HERE, digest, prepare, timestamp, validated_build
from pipeline import Program, compile_harness, execute, read_snapshot
from operator_source import source_key
from run_swiglu import REPO, checked_command, runtime_environment, torch, triton, write_json
from run_suite import check_case
from workloads import Workloads
import triton.language as tl


@triton.jit
def _matrix(A, B, C, D, K: tl.constexpr, HALF: tl.constexpr, INCLUDE: tl.constexpr,
            BK: tl.constexpr, TRANSPOSE: tl.constexpr, RESET: tl.constexpr):
    row = tl.arange(0, 64)
    col = tl.arange(0, 64)
    inner = tl.arange(0, BK)
    dtype: tl.constexpr = tl.float16 if HALF else tl.float32
    if INCLUDE:
        acc = tl.load(C + row[:, None] * 64 + col[None, :]).to(dtype)
    else:
        acc = tl.full((64, 64), 0, dtype)
    total = tl.full((64, 64), 0, tl.float32)
    for base in range(0, K, BK):
        a = tl.load(A + row[:, None] * K + base + inner[None, :])
        if TRANSPOSE:
            b = tl.load(B + col[None, :] * K + base + inner[:, None])
        else:
            b = tl.load(B + (base + inner[:, None]) * 64 + col[None, :])
        if HALF:
            a = a.to(tl.float16)
        if RESET:
            chunk = tl.dot(a, b, out_dtype=dtype)
            total += chunk.to(tl.float32)
        else:
            acc = tl.dot(a, b, acc, out_dtype=dtype)
    tl.store(D + row[:, None] * 64 + col[None, :], total if RESET else acc)


def matrix_program(half, blocks, include, *, bk=16, transpose=False, reset=False):
    k = blocks * 16
    # Small distinct integers: the exact expected answer does not depend on
    # an unspecified reduction order. Every row/column/fragment is checked.
    row, inner, col = torch.arange(64)[:, None], torch.arange(k)[None, :], torch.arange(64)[None, :]
    a = ((row * 3 + inner * 7 + row * inner) % 5 - 2).to(torch.bfloat16)
    b = ((torch.arange(k)[:, None] * 5 + col * 3 + torch.arange(k)[:, None] * col) % 7 - 3)
    dtype = torch.float16 if half else torch.float32
    b = b.to(torch.float16 if half else torch.bfloat16)
    c = ((row * 7 + col * 11) % 17 - 8).to(dtype)
    suffix = f"_bk{bk}_{'transposed' if transpose else 'contiguous'}{'_reset' if reset else ''}"
    p = Program(f"{'f16' if half else 'bf16'}_k{k}_{'accumulate' if include else 'overwrite'}{suffix}", "instruction_probe")
    stored_b = b.T.contiguous() if transpose else b
    refs = [p.buffer(n, x.contiguous(), readonly=True) for n, x in zip(("a", "b", "c"), (a, stored_b, c))]
    out_dtype = torch.float32 if reset else dtype
    out = p.output("output", (64, 64), out_dtype)
    p.launch(_matrix, dict(A=refs[0], B=refs[1], C=refs[2], D=out, K=k, HALF=half, INCLUDE=include,
                          BK=bk, TRANSPOSE=transpose, RESET=reset),
             (1,), num_warps=4, num_stages=3 if bk == 64 else 1)
    expected = (a.double() @ b.double() + (c.double() if include and not reset else 0)).to(out_dtype)
    return p, expected


@triton.jit
def _conversion(A, B, Out, N: tl.constexpr):
    i = tl.program_id(0) * 128 + tl.arange(0, 128)
    a, b = tl.load(A + i, i < N, 0), tl.load(B + i, i < N, 0)
    sa, sb, lo, hi = tl.inline_asm_elementwise(
        "{ .reg .b32 pair; cvt.rn.f16.f32 $0,$4; cvt.rn.f16.f32 $1,$5; "
        "cvt.rn.f16x2.f32 pair,$4,$5; mov.b32 {$2,$3},pair; }",
        "=h,=h,=h,=h,f,f", [a, b], (tl.float16, tl.float16, tl.float16, tl.float16),
        is_pure=True, pack=1)
    tl.store(Out + 4 * i, sa, i < N)
    tl.store(Out + 4 * i + 1, sb, i < N)
    tl.store(Out + 4 * i + 2, lo, i < N)
    tl.store(Out + 4 * i + 3, hi, i < N)


def conversion_program():
    # Midpoints exercise ties-even; adjacent FP32 values exercise either side.
    values = torch.arange(0x7bff, dtype=torch.int16).view(torch.float16).float()
    adjacent = torch.arange(1, 0x7c00, dtype=torch.int16).view(torch.float16).float()
    midpoint = ((values + adjacent) * 0.5)[::8]
    a = torch.cat([torch.nextafter(midpoint, values[::8]), midpoint,
                   torch.nextafter(midpoint, adjacent[::8]),
                   torch.tensor([0., -0., 65504., 65520., 1e-30])])
    a = torch.cat([a, -a])
    b = a.flip(0).contiguous()
    p = Program("fp16_conversions", "instruction_probe")
    ar, br = p.buffer("a", a, readonly=True), p.buffer("b", b, readonly=True)
    out = p.output("output", (a.numel(), 4), torch.float16)
    p.launch(_conversion, dict(A=ar, B=br, Out=out, N=a.numel()), (triton.cdiv(a.numel(), 128),))
    return p, torch.stack([a.half(), b.half(), b.half(), a.half()], 1)


def exact_check(actual, expected, name):
    # Instruction conformance includes signed zeros and expected infinities;
    # use bits, not the production checker's finite-only error metric.
    bits_dtype = torch.int16 if actual.element_size() == 2 else torch.int32
    mismatches = actual.contiguous().view(bits_dtype) != expected.contiguous().view(bits_dtype)
    return {"name": name, "comparison": "all_output_bits", "elements": actual.numel(),
            "mismatched_elements": int(mismatches.sum()),
            "nonfinite_expected_elements": int((~torch.isfinite(expected)).sum()),
            "passed": not bool(mismatches.any())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--suite", choices=("probes", "router", "production"), default="probes")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1108, 9073])
    args = ap.parse_args()
    torch.set_num_threads(1)
    root = prepare(args.root)
    run = root / "runs" / f"sim003-wgmma-{args.suite}-{timestamp()}"
    run.mkdir()
    os.environ["TRITON_CACHE_DIR"] = str(run / "triton-cache")
    source, library, env = runtime_environment(root)
    exe = compile_harness(run, library, env)
    report = {"experiment": "sim003", "kind": "router_wgmma", "suite": args.suite,
              "seeds": args.seeds, "timeout_seconds": args.timeout,
              "run_directory": str(run), "gpu_execution": False, "h20_tested": False,
              "build": validated_build(root, source, library), "cases": [],
              "accumulation_model": "ascending K FP32 sum, add C, FP16 RN per instruction; not H20 bitwise calibrated",
              "source_sha256": {source_key(p): digest(p) for p in
                  [*HERE.glob('*.py'), *HERE.glob('*.h'), HERE/'pipeline.cpp', HERE/'test_wgmma_numeric.cpp',
                   REPO/'Router_GEMM_prefill/solution.py', REPO/'Router_GEMM_prefill/benchmark.py']}}
    if args.suite == "probes":
        test = run / "numeric-test"
        checked_command(["g++", "-std=c++17", "-O2", f"-I{env['CUDA_INSTALL_PATH']}/include",
                         HERE / "test_wgmma_numeric.cpp", "-o", test], run, env, "compile-numeric.log")
        log, _ = checked_command([test], run, env, "numeric.log")
        if "WGMMA_NUMERIC_PASS" not in log.read_text():
            raise RuntimeError("Numeric unit test did not complete")
        report["numeric_test"] = log.read_text()
        entries = [(matrix_program(h, k, c), None) for h in (False, True) for k in (1, 4) for c in (False, True)]
        entries += [(matrix_program(h, 16, False, bk=64, transpose=t, reset=r), None)
                    for h in (False, True) for t in (False, True) for r in (False, True)]
        entries.append((conversion_program(), None))
    else:
        entries = []
        for seed in args.seeds:
            for tokens in ([2048] if args.suite == "production" else [2, 65]):
                case = Workloads(seed).router(f"router_{seed}_t{tokens}", tokens, packed_diagnostic=tokens < 2048)
                entries.append(((case.program, case.expected["output"]), case))
    for (p, expected), case in entries:
        folder = run / p.label
        folder.mkdir()
        row = {"name": p.label, "dispatch": p.dispatch, "artifacts": str(folder)}
        try:
            evidence, outputs = execute(p, folder, source, library, env, exe, args.timeout)
            row.update(evidence)
            ptx = "\n".join(f.read_text() for f in sorted(folder.glob('kernel*.ptx')))
            required = ["m64n64k16.f16.f16.f16"] if p.label.startswith('f16_') else ["m64n64k16.f32.bf16.bf16"]
            if p.label == "fp16_conversions":
                required = []
                if "cvt.rn.f16.f32" not in ptx or "cvt.rn.f16x2.f32" not in ptx:
                    raise RuntimeError("Compiler omitted conversion probe instructions")
            if case is not None:
                required = ["m64n64k16.f32.bf16.bf16", "m64n64k16.f16.f16.f16"]
            if any(f"wgmma.mma_async.sync.aligned.{s}" not in ptx for s in required):
                raise RuntimeError("Compiler did not emit the WGMMA variants under test")
            row["required_variants"] = required
            if case is not None:
                row.update(check_case(case, outputs, {}))
                w = p.buffers["w"]["tensor"]
                high = w.bfloat16()
                low = ((w - high.float()) * 256).half()
                for name, reference in (("high", high), ("low", low)):
                    packed = read_snapshot(folder / f"step000-{name}.bin", reference)
                    row["checks"][f"pack_{name}"] = exact_check(packed, reference, f"pack_{name}")
                    row["checks"][f"preserved_{name}"] = exact_check(outputs[name], packed, f"preserved_{name}")
                if not all(c["passed"] for c in row["checks"].values()):
                    row["status"] = "FUNCTIONAL_FAIL"
            else:
                check = exact_check(outputs["output"], expected, p.label)
                passed = check["passed"]
                row.update(status="FUNCTIONAL_PASS" if passed else "FUNCTIONAL_FAIL",
                           exact_bits_result=passed, check=check)
        except subprocess.TimeoutExpired as error:
            row.update(status="SIMULATION_TIMEOUT", error=str(error))
        except Exception as error:
            message = str(error)
            row.update(status="UNSUPPORTED" if "unsupported" in message.lower() else "EXECUTION_ERROR", error=message)
        if "completed_launches" not in row:
            text = (folder / "simulation.log").read_text(errors="replace") if (folder / "simulation.log").exists() else ""
            row["completed_launches"] = len(re.findall(r"(?m)^SIM002_STEP_COMPLETE\s+\d+", text))
        row["exports"] = [json.loads(p.read_text()) for p in sorted(folder.glob('export*.json'))]
        write_json(folder / "result.json", row)
        report["cases"].append(row)
        print(json.dumps({"name": p.label, "status": row["status"], "launches": row["completed_launches"],
                          "error": row.get("error", "")[-900:]}), flush=True)
    report["summary"] = {s: sum(r["status"] == s for r in report["cases"]) for s in sorted({r["status"] for r in report["cases"]})}
    report["status"] = "PASS" if all(r["status"] == "FUNCTIONAL_PASS" for r in report["cases"]) else "HAS_FAILURES"
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected physical CUDA initialization")
    output = HERE / f"results-sim003-wgmma-{args.suite}-{timestamp()[:15]}.json"
    write_json(output, report)
    print(json.dumps({"report": str(output), "summary": report["summary"]}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
