import argparse
import json
import statistics
from datetime import datetime, timezone
from time import perf_counter_ns

import torch
import torch.nn.functional as functional

from verify import DAY, check_output, compile_solution, make_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    torch.manual_seed(131117)
    major, minor = torch.cuda.get_device_capability()
    implementations = {}
    hashes = {}
    for name in ("pool-small-tuned.cu", "pool-views.cu"):
        implementations[name], hashes[name] = compile_solution(
            1, DAY / "1" / name, args.nvcc, f"{major}{minor}"
        )
    cases = []
    for n, k, s, p in ((192, 5, 2, 2), (224, 7, 3, 3), (512, 3, 3, 1), (784, 5, 2, 2), (1024, 3, 3, 1)):
        for shift in (1, 2, 3):
            x = make_input((n, n, n), shift)
            ref = functional.avg_pool3d(x[None, None], k, s, p, count_include_pad=True)[0, 0]
            arguments = lambda out: (x.data_ptr(), k, s, p, out, n, n, n)
            errors = {
                name: check_output(arguments, f, ref.shape, ref, 2e-6, 1e-5, start=shift)
                for name, f in implementations.items()
            }
            case = {"size": n, "start_floats": shift, "max_abs": errors}
            if shift == 1:
                y = torch.empty(ref.numel()+shift, device="cuda")[shift:]
                args = arguments(y.data_ptr())
                times = {name: [] for name in implementations}
                names = list(implementations)
                for trial in range(7):
                    for name in names[::1 if trial % 2 == 0 else -1]:
                        torch.cuda.synchronize()
                        start = perf_counter_ns()
                        implementations[name](*args)
                        torch.cuda.synchronize()
                        times[name].append((perf_counter_ns()-start)/1000)
                case["complete_median_us"] = {name: statistics.median(values) for name, values in times.items()}
                del y
            print(case, flush=True)
            cases.append(case)
            del x, ref
    record = {"gpu": torch.cuda.get_device_name(), "sha256": hashes, "cases": cases}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    path = DAY / f"results-views-{stamp}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
