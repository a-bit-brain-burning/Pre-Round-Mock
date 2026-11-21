"""Run CPU-only checker, replay-boundary and BF16 rounding regression tests."""

import argparse
from pathlib import Path
import sys

from bootstrap import DEFAULT_ROOT, HERE, digest, prepare, timestamp, validated_build
from run_swiglu import REPO, checked_command, runtime_environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("p1", "p2"), default="p1")
    args = parser.parse_args()
    root = prepare(args.root)
    source, library, env = runtime_environment(root)
    folder = root / "runs" / f"cpu-checks-{timestamp()}"
    folder.mkdir()
    env["PYTHONPATH"] = str(REPO)
    env["PREROUND_OPERATOR_ROOT"] = str(REPO)
    # Keep retained unit-test fixtures within the governed persistent workspace.
    env["TMPDIR"] = str(folder)
    logs = {}
    for name, command in (
        ("checker", [sys.executable, "-B", "-m", "unittest", "common.test_validation"]),
        ("adapter", [sys.executable, "-B", HERE / "test_adapter.py"]),
    ):
        log, _ = checked_command(command, folder, env, f"{name}.log")
        logs[name] = {"log": str(log), "output": log.read_text()}
    executable = folder / "bf16-conversion-test"
    checked_command(["g++", "-std=c++17", "-O2", f"-I{env['CUDA_INSTALL_PATH']}/include",
                     HERE / "test_bf16_conversion.cpp", "-o", executable], folder, env, "compile-bf16.log")
    log, _ = checked_command([executable], folder, env, "bf16.log")
    if "BF16_CONVERSION_PASS 393216 rounding cases; 65536 expansions" not in log.read_text():
        raise RuntimeError("BF16 conversion tests did not complete")
    logs["bf16"] = {"log": str(log), "output": log.read_text()}
    files = [HERE / "test_adapter.py", HERE / "test_bf16_conversion.cpp", HERE / "bf16_conversion.h", HERE / "run_checks.py"]
    if args.phase == "p2":
        log, _ = checked_command([sys.executable, "-B", HERE / "test_pipeline.py"], folder, env, "pipeline.log")
        logs["pipeline"] = {"log": str(log), "output": log.read_text()}
        executable = folder / "numeric-conversion-test"
        checked_command(["g++", "-std=c++17", "-O2", f"-I{env['CUDA_INSTALL_PATH']}/include",
                         HERE / "test_numeric_conversions.cpp", "-o", executable], folder, env, "compile-numeric.log")
        log, _ = checked_command([executable], folder, env, "numeric.log")
        if "NUMERIC_CONVERSIONS_PASS" not in log.read_text():
            raise RuntimeError("Numeric conversion tests did not complete")
        logs["numeric"] = {"log": str(log), "output": log.read_text()}
        files.extend(HERE / n for n in ("test_pipeline.py", "test_numeric_conversions.cpp", "numeric_conversions.h"))
    experiment = "sim002" if args.phase == "p2" else "sim001"
    report = {"experiment": experiment, "kind": "cpu_regression", "status": "PASS",
              "gpu_execution": False, "h20_tested": False, "run_directory": str(folder),
              "build": validated_build(root, source, library), "tests": logs,
              "source_sha256": {p.name: digest(p) for p in files}}
    output = args.output or HERE / f"results-{experiment}-checks-{timestamp()[:15]}.json"
    write_json(output, report)
    print(f"CPU_CHECKS_PASS; report: {output}")


if __name__ == "__main__":
    main()
