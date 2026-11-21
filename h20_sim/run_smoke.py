"""Build and run an independent CUDA SM90 vector-add through FlashGPU-Sim."""

import argparse
from pathlib import Path
import re

from bootstrap import DEFAULT_ROOT, HERE, LOCK, digest, prepare, timestamp, validated_build
from run_swiglu import checked_command, copy_config, runtime_environment, write_json, parse_resource_usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = prepare(args.root)
    folder = root / "runs" / f"cuda-smoke-{timestamp()}"
    folder.mkdir()
    source, library, env = runtime_environment(root)
    copy_config(source, folder)
    cuda = Path(env["CUDA_INSTALL_PATH"])
    for kind in ("ptx", "cubin"):
        checked_command([cuda / "bin/nvcc", "-std=c++17", "-arch=sm_90", f"--{kind}",
                         HERE / "smoke.cu", "-o", folder / f"smoke.{kind}"], folder, env, f"compile-{kind}.log")
    resources, _ = checked_command(["cuobjdump", "--dump-resource-usage", folder / "smoke.cubin"],
                                    folder, env, "resources.log")
    allocation = next(r for r in parse_resource_usage(resources.read_text()) if r["function"] == "vector_add")
    (folder / "smoke.ptxinfo").write_text(
        "ptxas info    : 0 bytes gmem\n"
        "ptxas info    : Compiling entry function 'vector_add' for 'sm_90'\n"
        "ptxas info    : Function properties for vector_add\n"
        f"    {allocation.get('stack', 0)} bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
        f"ptxas info    : Used {allocation['reg']} registers, {allocation.get('shared', 0)} bytes smem, 0 bytes cmem[0], 0 bytes lmem\n")
    checked_command(["g++", "-x", "c++", "-std=c++17", "-DH20_SIM_HOST", f"-I{cuda}/include",
                     HERE / "smoke.cu", f"-L{library.parent}", f"-Wl,-rpath,{library.parent}",
                     "-lcudart", "-o", folder / "smoke"], folder, env, "compile-host.log")
    linkage, _ = checked_command(["ldd", folder / "smoke"], folder, env, "linkage.log")
    if str(library.parent) not in linkage.read_text() or "/usr/lib/wsl" in linkage.read_text():
        raise RuntimeError("Smoke program is not linked to the simulator runtime")
    log, seconds = checked_command(["stdbuf", "-oL", "-eL", folder / "smoke"], folder, env, "simulation.log")
    text = log.read_text(errors="replace")
    if "SM90_VECTOR_ADD_PASS 37/37" not in text:
        raise RuntimeError("SM90 vector-add numerical check did not pass")
    cycles = re.findall(r"(?m)^gpu_tot_sim_cycle\s*=\s*(\d+)", text)
    report = {"experiment": "sim001", "kind": "cuda_sm90_smoke", "status": "PASS",
              "elements_verified": 37, "run_directory": str(folder), "backend": LOCK,
              "build": validated_build(root, source, library),
              "source_sha256": digest(HERE / "smoke.cu"), "library_sha256": digest(library),
              "simulated_cycles_upstream_h100": int(cycles[-1]) if cycles else None,
              "simulator_wall_seconds": seconds, "gpu_execution": False, "h20_tested": False}
    output = args.output or HERE / f"results-sim001-cuda-smoke-{timestamp()[:15]}.json"
    write_json(output, report)
    print(f"SM90_VECTOR_ADD_PASS 37/37; report: {output}")


if __name__ == "__main__":
    main()
