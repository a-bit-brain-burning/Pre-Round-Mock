import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as functional


DAY = Path(__file__).resolve().parent
CACHE = DAY.parent / ".cache" / "Day2"
PTR = ctypes.c_void_p
INT = ctypes.c_int
SIZE = ctypes.c_size_t
SIGNATURES = {
    1: [PTR, INT, INT, INT, PTR, SIZE, SIZE, SIZE],
    2: [PTR, PTR, SIZE, SIZE, SIZE, SIZE],
    3: [PTR, INT, PTR, SIZE, SIZE],
}


def compile_solution(number, source, nvcc, capability):
    CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = CACHE / f"day2-{number}-{digest[:12]}-sm{capability}.so"
    command = [nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler=-fPIC"]
    for arch in sorted({"90", capability}):
        command.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    subprocess.run(
        [*command, "-x", "cu", str(source), "-o", str(target)],
        check=True, env={**os.environ, "TMPDIR": str(CACHE)},
    )
    function = ctypes.CDLL(str(target)).solution
    function.argtypes = SIGNATURES[number]
    function.restype = None
    return function, digest


def make_input(shape, shift=0):
    count = 1
    for n in shape:
        count *= n
    data = torch.randn(count + shift, device="cuda", dtype=torch.float32)
    return data[shift:].view(shape)


def check_output(function_args, function, shape, expected, atol, rtol, start=None, tensor_output=False):
    # 两端保留哨兵，输出区域填入 NaN，检查边界写入和遗漏的元素。
    if start is None:
        start = 4 if expected.numel() >= 1 << 20 else 1
    out = torch.full((start + expected.numel() + 1,), float("nan"), device="cuda")
    sentinel = -1234567.5
    out[:start] = out[-1] = sentinel
    result = out[start:-1].view(shape)
    function(*function_args(result if tensor_output else result.data_ptr()))
    torch.cuda.synchronize()
    if not (out[:start] == sentinel).all().item() or out[-1].item() != sentinel:
        raise AssertionError("输出数组两端的哨兵发生变化")
    maximum = torch.zeros((), device="cuda")
    for actual, ref in zip(result.flatten().split(1 << 20), expected.flatten().split(1 << 20)):
        torch.testing.assert_close(actual, ref, atol=atol, rtol=rtol)
        maximum = torch.maximum(maximum, (actual - ref).abs().max())
    return maximum.item()


def check_pool(function, tensors=False):
    cases = [
        ((1, 1, 1), 1, 1, 0), ((7, 9, 17), 2, 1, 0),
        ((10, 14, 103), 3, 3, 0), ((16, 13, 97), 3, 3, 1),
        ((11, 17, 37), 5, 2, 0), ((13, 9, 81), 5, 2, 2),
        ((17, 18, 101), 7, 3, 2), ((9, 11, 12), 4, 2, 2),
        ((192,) * 3, 5, 2, 2), ((224,) * 3, 7, 3, 3), ((512,) * 3, 3, 3, 1),
        ((784,) * 3, 5, 2, 2), ((1024,) * 3, 3, 3, 1),
    ]
    for shape, k, stride, padding in cases:
        x = make_input(shape, shift=int(shape[0] < 100))
        if shape == (13, 9, 81):
            x.fill_(1)
        expected = functional.avg_pool3d(
            x[None, None], k, stride, padding, count_include_pad=True,
        )[0, 0]
        starts = (4, 1) if shape[0] in (192, 224) else (4,) if shape[0] >= 100 else (1,)
        for start in starts:
            error = check_output(lambda out: (x if tensors else x.data_ptr(), k, stride, padding, out, *shape),
                                 function, expected.shape, expected, 2e-6, 1e-5, start=start, tensor_output=tensors)
            yield {"shape": shape, "kernel": k, "stride": stride, "padding": padding,
                   "output_start": start, "max_abs": error}
        del x, expected


def check_norm(function):
    cases = [
        ((1, 2, 3, 5), "axes"), ((3, 5, 7, 13), "random"),
        ((4, 7, 3, 129), "constant"), ((5, 3, 11, 191), "large_mean"),
        ((4, 4, 256, 256), "isolated"), ((2, 2, 512, 512), "wide_values"),
        ((16, 64, 256, 256), "random"), ((32, 128, 128, 128), "random"),
        ((8, 256, 64, 64), "random"), ((4, 32, 512, 512), "random"),
    ]
    for shape, kind in cases:
        x = make_input(shape, shift=int(shape[3] % 2 != 0))
        if kind == "constant":
            x.fill_(3.25)
        if kind == "large_mean":
            x.mul_(0.05).add_(10000)
        if kind == "isolated":
            x.mul_(0.001).add_(10000)
            x[0, :, 0, 0] = 10010
        if kind == "wide_values":
            x.mul_(1e25)
        if kind in ("constant", "large_mean", "isolated", "wide_values"):
            data = x.double()
            var, mean = torch.var_mean(data, dim=(0, 2, 3), correction=0, keepdim=True)
            expected = ((data - mean) * torch.rsqrt(var + 1e-5)).float()
        else:
            expected = functional.batch_norm(x, None, None, training=True, eps=1e-5)
        error = check_output(lambda out: (x.data_ptr(), out, *shape),
                             function, shape, expected, 4e-6, 1e-5)
        record = {"shape": shape, "input": kind, "max_abs": error}
        if kind == "axes":
            var, mean = torch.var_mean(x, dim=0, correction=0, keepdim=True)
            batch_only = (x - mean) * torch.rsqrt(var + 1e-5)
            gap = (batch_only - expected).abs().max().item()
            assert gap > 0.5, "两种统计维度应当产生不同输出"
            record["batch_only_vs_reference_max_abs"] = gap
        yield record


def check_blur(function):
    cases = [
        ((1, 1), 1), ((1, 17), 15), ((5, 3), 27), ((19, 101), 3),
        ((35, 109), 15), ((17, 93), 21), ((33, 197), 27), ((15, 8), 31),
        ((1920, 1080), 15), ((1920, 1080), 21), ((1920, 1080), 27),
        ((2048, 2048), 15), ((2048, 2048), 21), ((2048, 2048), 27),
        ((1080, 1920), 21),
    ]
    for shape, k in cases:
        x = make_input(shape, shift=int(shape[0] < 100))
        if shape == (5, 3):
            x.fill_(1)
        expected = functional.avg_pool2d(
            x[None, None], k, stride=1, padding=k // 2, count_include_pad=False,
        )[0, 0]
        error = check_output(lambda out: (x.data_ptr(), k, out, *shape),
                             function, shape, expected, 4e-6, 2e-5)
        yield {"shape": shape, "kernel": k, "max_abs": error}
        del x, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--problems", type=int, choices=(1, 2, 3), nargs="+", default=[1, 2, 3])
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.source is not None and len(args.problems) != 1:
        parser.error("--source 需要指定一道题的 --problems 编号")
    torch.manual_seed(1117)
    major, minor = torch.cuda.get_device_capability()
    report = {
        "utc": datetime.now(timezone.utc).isoformat(), "seed": 1117,
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "nvcc": subprocess.check_output([args.nvcc, "--version"], text=True).strip(),
        "problems": {},
    }
    checks = {1: check_pool, 2: check_norm, 3: check_blur}
    for number in args.problems:
        source = args.source.resolve() if args.source else DAY / str(number) / "code.cpp"
        if source.suffix == ".py":
            if number != 1:
                parser.error("Triton 验证入口目前支持第 1 题")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            spec = importlib.util.spec_from_file_location("pool_" + digest[:12], source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            experiment = check_pool(module.solution, tensors=True)
        else:
            function, digest = compile_solution(number, source, args.nvcc, f"{major}{minor}")
            experiment = checks[number](function)
        cases = []
        for case in experiment:
            print(f"PASS {number} {case}", flush=True)
            cases.append(case)
        report["problems"][str(number)] = {"source": source.relative_to(DAY).as_posix(), "sha256": digest, "cases": cases}
        torch.cuda.empty_cache()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    path = DAY / f"results-local-{stamp}.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(path)


if __name__ == "__main__":
    main()
