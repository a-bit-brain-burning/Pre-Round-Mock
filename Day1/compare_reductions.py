import argparse
from datetime import datetime, timezone
import hashlib
import json
from statistics import median
import subprocess

import torch

from verify import BUILD, HERE, SHAPES, SOURCES, build


def compare(nvcc, baseline):
    root = HERE.parent
    commit = subprocess.check_output(["git", "rev-parse", "--verify", baseline + "^{commit}"],
                                     cwd=root, text=True).strip()
    arch = "".join(map(str, torch.cuda.get_device_capability()))
    operations = ("argmax", "argmin")
    report = {"gpu": torch.cuda.get_device_name(), "sm": arch, "torch": str(torch.__version__),
              "baseline": commit, "seed": 1116, "batch": 8, "warmup": 16, "rounds": 16,
              "nvcc": subprocess.check_output([nvcc, "--version"], text=True).strip(),
              "sha256": {}, "cases": []}
    torch.manual_seed(report["seed"])
    BUILD.mkdir(parents=True, exist_ok=True)
    for op in operations:
        path = SOURCES[op].relative_to(root).as_posix()
        previous = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
        report["sha256"][path] = hashlib.sha256(SOURCES[op].read_bytes()).hexdigest()
        versions = {"baseline": build(nvcc, {arch, "90"}, (op, "baseline", []), previous)[2],
                    "current": build(nvcc, {arch, "90"}, (op, "current", []))[2]}
        for shape, dim in SHAPES:
            data = torch.randn(shape, device="cuda")
            sizes = torch.tensor(shape, device="cuda", dtype=torch.int32)
            expected = getattr(torch, op)(data, dim=dim).int()
            out = torch.empty_like(expected)
            args = (data.data_ptr(), dim, out.data_ptr(), sizes.data_ptr(), len(shape))
            for fn in versions.values():
                out.fill_(-1)
                fn(*args)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
            for _ in range(report["warmup"]):
                for fn in versions.values():
                    fn(*args)
            torch.cuda.synchronize()
            times = {name: [] for name in versions}
            for i in range(report["rounds"]):
                order = tuple(versions) if i % 2 == 0 else tuple(reversed(versions))
                for name in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(report["batch"]):
                        versions[name](*args)
                    end.record()
                    end.synchronize()
                    times[name].append(start.elapsed_time(end) / report["batch"])
            medians = {name: median(samples) for name, samples in times.items()}
            report["cases"].append(dict(op=op, shape=shape, dim=dim, ms=times, median_ms=medians))
            print(op, shape, medians, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--baseline", default="90d676f334853e960df410431bb8eb24656ed24f")
    args = parser.parse_args()
    report = compare(args.nvcc, args.baseline)
    now = datetime.now(timezone.utc)
    report["utc"] = now.isoformat()
    path = HERE / f"results-1116-reductions-{now:%Y%m%d-%H%M%S}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(path)
