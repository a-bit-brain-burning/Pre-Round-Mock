"""Validate paired launch configurations on identical CPU inputs, five tasks only."""

import argparse
import fnmatch
import json
import os
from pathlib import Path
import subprocess
import tempfile

from bootstrap import DEFAULT_ROOT, HERE, digest, prepare, timestamp, validated_build
from operator_source import report_directory, source_key
from pipeline import compile_harness, execute
from run_suite import check_case
from run_swiglu import load_module, runtime_environment, torch, triton, write_json
from workloads import FOLDERS, REPO, Workloads

TASKS = ("swiglu", "gqa", "sink", "moe")


def candidates(task, seed):
    for variant in (0, 1):
        w = Workloads(seed)
        label = f"{task}_seed{seed}_variant{variant}"
        if task == "swiglu":
            case = w.swiglu_bf16(label, (32, 1024), (256, 1024)[variant])
        elif task == "gqa":
            case = w.gqa(label, [w.request(2, 65, torch.bfloat16)])
            if variant == 0:
                case.program.launches[0]["arguments"]["BN"] = 32
                case.program.dispatch = "diagnostic_bn32"
        elif task == "sink":
            case = w.sink(label, torch.bfloat16, tiled=True)
            if variant == 0:
                p = case.program
                launch = p.launches[0]
                a = launch["arguments"]
                names = w.modules["sink"]._sink.arg_names
                p.launches = []
                p.launch(w.modules["sink"]._sink, {n: a[n] for n in names}, (4, 4), num_warps=4)
                p.dispatch = "diagnostic_scalar_on_tiled_shape"
        elif task == "moe":
            case = w.moe(label, torch.bfloat16, grouped=bool(variant))
        elif task == "router":
            case = w.router(label, 2, packed_diagnostic=True)
            case.program.launches[1]["options"]["num_stages"] = (1, 3)[variant]
        else:
            raise ValueError("Only the five contest tasks are supported")
        case.metadata.update(comparison_group=f"{task}_seed{seed}", variant=variant,
                             comparison_scope="these inputs and unchanged source only; not full contest validation")
        yield case


def spill_candidates(task, seed):
    """Same original kernels, inputs and checkers; only launch constants differ."""
    specs = [(13, "full", 0, [(2, 7), (4, 130)]),
             (128, "full", 0, [(2, 32), (2, 129)]),
             (13, "sliding", 33, [(1, 33), (2, 66), (1, 129)]),
             (128, "sliding", 17, [(2, 33)])]
    if task == "gqa":
        for d, mode, window, shapes in specs:
            for bn in (32, 128):
                w = Workloads(seed)
                group = f"gqa_d{d}_{mode}_seed{seed}"
                case = w.gqa(f"{group}_bn{bn}", [w.request(q, k, torch.bfloat16, d=d) for q, k in shapes],
                             mode=mode, window=window or 17)
                original = case.program.launches[0]["arguments"]["BN"]
                case.program.launches[0]["arguments"]["BN"] = bn
                if bn != original:
                    case.program.dispatch = f"diagnostic_bn{bn}"
                case.metadata.update(comparison_group=group, block_n=bn, comparison_scope="validated inputs only")
                yield case
    elif task == "router":
        for tokens in (2, 65, 2048):
            for bk in (32, 64):
                w = Workloads(seed)
                group = f"router_t{tokens}_seed{seed}"
                case = w.router(f"{group}_bk{bk}", tokens, packed_diagnostic=True)
                case.program.launches[1]["arguments"]["BK"] = bk
                if bk != 64:
                    case.program.dispatch = f"diagnostic_packed_bk{bk}"
                case.metadata.update(comparison_group=group, block_k=bk, comparison_scope="validated inputs only")
                yield case


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--tasks", nargs="+", choices=(*TASKS, "router"))
    ap.add_argument("--seed", type=int, default=1108)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--phase", choices=("sim004", "sim005", "sim006"), default="sim004")
    ap.add_argument("--case-module", type=Path, help="sim006 case definitions in csxsl/h20_sim (default: optimization_cases.py)")
    ap.add_argument("--case", default="*")
    args = ap.parse_args()
    if args.case_module is not None and (args.phase != "sim006" or args.case_module.parent != Path(".")
                                          or args.case_module.suffix != ".py"):
        ap.error("--case-module requires sim006 and a Python filename in csxsl/h20_sim")
    if args.phase == "sim006" and args.tasks and "router" in args.tasks:
        ap.error("sim006 covers the four non-GEMM tasks")
    tasks = args.tasks or (TASKS if args.phase == "sim006" else (*TASKS, "router"))
    output_dir = report_directory(args.phase)
    if args.phase == "sim006":
        definitions = output_dir / (args.case_module or "optimization_cases.py")
        generate = load_module("csxsl_" + definitions.stem, definitions).optimization_candidates
    else:
        generate = candidates if args.phase == "sim004" else spill_candidates
    torch.set_num_threads(1)
    root = prepare(args.root)
    run = Path(tempfile.mkdtemp(prefix=f"{args.phase}-candidates-{timestamp()}-", dir=root / "runs"))
    os.environ["TRITON_CACHE_DIR"] = str(run / "triton-cache")
    source, library, env = runtime_environment(root)
    exe = compile_harness(run, library, env)
    paths = [HERE / n for n in ("run_candidates.py", "pipeline.py", "pipeline.cpp", "workloads.py", "run_suite.py", "run_swiglu.py", "operator_source.py")]
    if args.phase == "sim006":
        paths.append(definitions)
    for folder in [*FOLDERS.values(), "SwiGLU_backward"]:
        paths += [p for n in ("solution.py", "reference.py") if (p := REPO / folder / n).exists()]
    paths += [REPO / "Router_GEMM_prefill/benchmark.py", REPO / "common/validation.py"]
    report = {"experiment": args.phase, "kind": "functional_candidate_pairs", "run_directory": str(run),
              "torch": torch.__version__, "triton": triton.__version__,
              "build": validated_build(root, source, library), "gpu_execution": False, "h20_tested": False,
              "source_sha256": {source_key(p): digest(p) for p in paths}, "cases": []}
    if args.phase == "sim006":
        report["case_module"] = source_key(definitions)
    invariant_outputs = {}
    prior_status = {}
    for task in tasks:
        for case in generate(task, args.seed):
            if not fnmatch.fnmatchcase(case.name, args.case):
                continue
            folder = run / case.name
            folder.mkdir()
            row = {"name": case.name, "seed": args.seed, "task": task, "metadata": case.metadata,
                   "dispatch": case.program.dispatch, "artifacts": str(folder)}
            try:
                evidence, outputs = execute(case.program, folder, source, library, env, exe, args.timeout)
                row.update(evidence)
                row.update(check_case(case, outputs, invariant_outputs))
                if required := case.metadata.get("required_passes"):
                    passed = all(prior_status.get(name) == "FUNCTIONAL_PASS" for name in required)
                    row["checks"]["prerequisites"] = {"passed": passed, "cases": required}
                    if not passed:
                        row["status"] = "FUNCTIONAL_FAIL"
            except subprocess.TimeoutExpired as error:
                row.update(status="SIMULATION_TIMEOUT", error=str(error))
            except Exception as error:
                row.update(status="EXECUTION_ERROR", error=f"{type(error).__name__}: {error}")
            write_json(folder / "result.json", row)
            prior_status[case.name] = row["status"]
            report["cases"].append(row)
            print(json.dumps({"case": case.name, "status": row["status"], "error": row.get("error", "")[-1000:]}), flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected physical CUDA initialization")
    report["summary"] = {s: sum(r["status"] == s for r in report["cases"]) for s in sorted({r["status"] for r in report["cases"]})}
    report["status"] = "PASS" if all(r["status"] == "FUNCTIONAL_PASS" for r in report["cases"]) else "HAS_FAILURES"
    if not report["cases"]:
        raise ValueError("No matching candidate cases")
    kind = f"candidates-seed{args.seed}" if args.phase == "sim006" else "candidates"
    stamp = timestamp()[:15]
    output = output_dir / f"results-{args.phase}-{kind}-{stamp}.json"
    try:
        write_json(output, report)
    except FileExistsError:
        run_id = run.name.rsplit("-", 1)[-1]
        output = output_dir / f"results-{args.phase}-{kind}-{run_id}-{stamp}.json"
        write_json(output, report)
    print(json.dumps({"report": str(output), "summary": report["summary"]}), flush=True)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
