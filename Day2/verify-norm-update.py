import argparse
from pathlib import Path

import torch
from torch.nn import functional

from verify import compile_solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    torch.manual_seed(117)
    major, minor = torch.cuda.get_device_capability()
    function, digest = compile_solution(
        2, args.source.resolve(), args.nvcc, f"{major}{minor}"
    )
    for b, f, side in ((8, 256, 64), (16, 64, 256), (32, 128, 128), (4, 32, 512)):
        x = torch.randn((b, f, side, side), device="cuda")
        y = torch.empty_like(x)
        for iteration in range(5):
            if iteration==1:
                x.neg_()
            if iteration==3:
                x[0,0,0,0]+=2
            if iteration==4:
                x=x.clone()
            ref = functional.batch_norm(x, None, None, training=True, eps=1e-5)
            y.fill_(float("nan"))
            function(x.data_ptr(), y.data_ptr(), *x.shape)
            torch.cuda.synchronize()
            for actual, expected in zip(y.flatten().split(1<<20), ref.flatten().split(1<<20)):
                torch.testing.assert_close(actual, expected, atol=4e-6, rtol=1e-5)
            print(digest[:12], tuple(x.shape), iteration, "PASS", flush=True)


if __name__ == "__main__":
    main()
