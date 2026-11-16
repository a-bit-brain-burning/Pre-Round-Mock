import argparse
import ctypes
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import subprocess

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import torch


HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / ".cache" / "Day1" / "build"
SOURCES = {
    "floyd_warshall": HERE / "1" / "code.cpp",
    "argmax": HERE / "2" / "code.cpp",
    "argmin": HERE / "3" / "code.cpp",
}
GRAPH_SIZES = (512, 1024, 2048, 4096)
SHAPES = (
    ((16, 128, 256), 1), ((32, 512, 512), 0), ((8, 1024, 1024), 2),
    ((64, 128, 128, 128), 2), ((4, 256, 256, 256), 1), ((128, 64, 64, 64), 3),
)


def configurations():
    for tile in (32, 64):
        for dpx in (0, 1):
            yield "floyd_warshall", f"tile{tile}_dpx{dpx}", [f"-DFW_TILE={tile}", f"-DFW_DPX={dpx}"]
    for operation in ("argmax", "argmin"):
        yield operation, "default", []
        yield operation, "scalar128", ["-DARG_COLUMNS_128=1"]
        yield operation, "scalar256", ["-DARG_COLUMNS_256=1"]
        yield operation, "warp1024", ["-DARG_LONG_WARPS=1"]


def build(nvcc, arches, config, source_bytes=None):
    operation, label, flags = config
    library = BUILD / f"{operation}_{label}.so"
    command = [nvcc, "-O3", "-std=c++17", "--shared", "-Xcompiler=-fPIC"]
    for arch in sorted(arches):
        command.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    source = str(SOURCES[operation]) if source_bytes is None else "-"
    command.extend([*flags, "-x", "cu", source, "-o", str(library)])
    subprocess.run(command, input=source_bytes, check=True, env={**os.environ, "TMPDIR": str(BUILD)})
    function = ctypes.CDLL(str(library)).solution
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    function.argtypes = ([ptr, ptr, ctypes.c_size_t] if operation == "floyd_warshall"
                         else [ptr, integer, ptr, ptr, integer])
    function.restype = None
    return operation, label, function


def make_graph(n):
    rng = np.random.default_rng(1116 + n)
    if n == 137:
        return rng.integers(0, 128, (n, n)).astype(np.float32)
    graph = np.zeros((n, n), dtype=np.float32)
    order = rng.permutation(n)
    # 随机排列顶点，使路径横跨多个方块，同时保留互不可达的分量。
    for group in np.array_split(order, 3):
        for shift in (1, 7, 19):
            graph[group, np.roll(group, shift)] = rng.integers(1, 256, len(group))
    np.fill_diagonal(graph, 0)
    return graph


def wide_graph():
    graph = np.zeros((193, 193), dtype=np.float32)
    graph[2, 66] = 2**23
    graph[66, 130] = 2**23
    graph[130, 132] = 256
    graph[3, 67] = 2**23 + 16
    graph[67, 131] = 32
    graph[4, 68] = 2**30
    graph[68, 133] = 1024
    graph[5, 69] = 1
    graph[69, 134] = 1
    graph[5, 134] = 2**30
    return graph


def record_case(operation, case, functions, call_args, out, expected, work, runs, records):
    for label, function in functions[operation]:
        out.fill_(-7)
        function(*call_args)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=0, atol=0, msg=None)
        if runs:
            samples = []
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(runs):
                start.record()
                function(*call_args)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            ms = median(samples)
            rate = ({"billion_updates_per_s": work / (ms * 1e6)} if operation == "floyd_warshall"
                    else {"input_TB_per_s": work / (ms * 1e9)})
            records.append(dict(operation=operation, case=case, config=label,
                                cuda_event_ms=samples, median_ms=ms, **rate))
    print(f"PASS {operation} {case} ({len(functions[operation])} configurations)", flush=True)


def check_graphs(functions, runs, records):
    for n in (1, 31, 33, 65, 129, 137, 193, *GRAPH_SIZES):
        graph = wide_graph() if n == 193 else make_graph(n)
        ref = dijkstra(csr_matrix(graph), directed=True)
        ref[np.isinf(ref)] = -1
        expected = torch.from_numpy(ref.astype(np.float32)).cuda()
        data = torch.from_numpy(graph).cuda()
        out = torch.empty_like(data)
        record_case("floyd_warshall", n, functions, (data.data_ptr(), out.data_ptr(), n),
                    out, expected, n**3, runs if n in GRAPH_SIZES else 0, records)


def check_indices(operation, case, functions, data, dim, expected, runs, records):
    out = torch.empty_like(expected)
    for device in ("cpu", "cuda"):
        sizes = torch.tensor(data.shape, device=device, dtype=torch.int32)
        args = (data.data_ptr(), dim, out.data_ptr(), sizes.data_ptr(), data.ndim)
        record_case(operation, f"{case},shape_device={device}", functions, args, out, expected,
                    data.numel() * data.element_size(), runs, records)


def check_reductions(functions, runs, records):
    extra = (((1,), 0), ((8, 8), 0), ((4, 65), 1), ((1097, 97), -1), ((7, 33, 5), 1),
             ((2, 5, 1), 1), ((1, 128, 128), 1), ((1, 256, 256, 256), 1))
    for shape, dim in (*extra, *SHAPES):
        public_case = (shape, dim) in SHAPES
        if public_case:
            data = torch.randn(shape, device="cuda")
        else:
            # 一个元素的起始位移同时检查 float2 地址要求。
            data = torch.empty(int(np.prod(shape)) + 1, device="cuda")[1:].view(shape)
            data.random_(-3, 4)
            data.select(dim, 0).fill_(float("nan"))
            if shape[dim] > 1:
                data.select(dim, -1).fill_(float("nan"))
                # 同一切片中同时出现首次极值、全相等数值和重复 NaN。
                flat = data.reshape(-1)
                flat[::3] = float("inf")
                flat[1::5] = -float("inf")
                flat[2::7] = -0.0
        for operation in ("argmax", "argmin"):
            expected = getattr(torch, operation)(data, dim=dim).int()
            check_indices(operation, f"{shape},dim={dim}", functions, data, dim, expected,
                          runs if public_case else 0, records)
    for operation in ("argmax", "argmin"):
        for length in (64, 1024):
            data = torch.zeros((3, length), device="cuda")
            value = 1 if operation == "argmax" else -1
            data[1, [31, 32, length - 1]] = value
            data[2, [-1]] = value
            expected = torch.tensor([0, 31, length - 1], dtype=torch.int32, device="cuda")
            check_indices(operation, f"ties-{length}", functions, data, 1, expected, 0, records)
            data = torch.full((7, length), -float("inf") if value == 1 else float("inf"),
                              device="cuda")
            data[1, [3, 4, length - 1]] = float("nan")
            data[2].fill_(-0.0)
            data[2, 1::2] = 0.0
            data[3, [5, length - 1]] = -value
            data[4, [1, 31, length - 2]] = float("inf") * value
            data[5, [0, 4, 33]] = -float("nan")
            data[6, length - 2] = value
            expected = getattr(torch, operation)(data, dim=1).int()
            check_indices(operation, f"vector-special-{length}", functions, data, 1, expected, 0, records)
        data = torch.zeros((3, 128, 256), device="cuda")
        data[:, [3, 4], ::2] = float("nan")
        data[:, [1, 4, 127], 1::2] = 1 if operation == "argmax" else -1
        expected = getattr(torch, operation)(data, dim=1).int()
        check_indices(operation, "column-groups", functions, data, 1, expected, 0, records)
        for length, inner in ((128, 128), (256, 65536)):
            data = torch.full((length, inner), -0.0, device="cuda")
            data[-1, ::4] = 0.0
            data[[5, length // 2, length - 1], 1::4] = float("nan")
            data[[7, length - 2], 2::4] = float("inf") if operation == "argmax" else -float("inf")
            data[length - 1, 3::4] = 3 if operation == "argmax" else -3
            expected = getattr(torch, operation)(data, dim=0).int()
            check_indices(operation, f"tiled-{length}", functions, data, 0, expected, 0, records)
        for offset in (0, 1):
            data = torch.empty(32 * 262144 + offset, device="cuda")[offset:].view(32, 262144)
            data.zero_()
            data[[2, 8, 31], 0::4] = float("nan")
            data[[3, 17], 1::4] = 1 if operation == "argmax" else -1
            data[31, 2::4] = 2 if operation == "argmax" else -2
            expected = getattr(torch, operation)(data, dim=0).int()
            check_indices(operation, f"columns32-offset-{offset}", functions, data, 0, expected, 0, records)
        data = torch.randn((8, 8), device="cuda")
        sizes = torch.tensor(data.shape, dtype=torch.int32, device="cpu", pin_memory=True)
        expected = getattr(torch, operation)(data, dim=0).int()
        out = torch.empty_like(expected)
        record_case(operation, "pinned-shape", functions,
                    (data.data_ptr(), 0, out.data_ptr(), sizes.data_ptr(), 2), out, expected, 0, 0, records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--timing-runs", type=int, default=0)
    args = parser.parse_args()
    if args.timing_runs < 0:
        parser.error("--timing-runs must be nonnegative")
    torch.manual_seed(1116)
    major, minor = torch.cuda.get_device_capability()
    arches = {"90", f"{major}{minor}"}
    BUILD.mkdir(parents=True, exist_ok=True)
    records = []
    functions = {"floyd_warshall": [], "argmax": [], "argmin": []}
    print(f"Building SM {sorted(arches)}; testing {torch.cuda.get_device_name()}", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for operation, label, function in pool.map(lambda config: build(args.nvcc, arches, config), configurations()):
            functions[operation].append((label, function))
    check_graphs(functions, args.timing_runs, records)
    check_reductions(functions, args.timing_runs, records)
    if records:
        now = datetime.now(timezone.utc)
        report = {
            "utc": now.isoformat(), "gpu": torch.cuda.get_device_name(), "sm": [major, minor],
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "seed": 1116,
            "nvcc": subprocess.check_output([args.nvcc, "--version"], text=True).strip(),
            "sha256": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(SOURCES.values())},
            "samples": records,
        }
        path = HERE / f"results-1116-gpu-{now:%Y%m%d-%H%M%S}.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(path)


if __name__ == "__main__":
    main()
