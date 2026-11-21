"""Typed, auditable single-process SM90 execution with persistent buffers."""

from dataclasses import dataclass
import math
from pathlib import Path
import re
import struct

from bootstrap import HERE, digest
from run_swiglu import (ASTSource, GPUTarget, UnsupportedKernel, checked_command,
                        copy_config, instruction_counts, parse_resource_usage,
                        raw_tensor, torch, triton, write_json)

GUARD = b"\xa5" * 128
TYPES = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16",
         torch.float64: "fp64", torch.int32: "i32", torch.int64: "i64",
         torch.float8_e4m3fn: "fp8e4nv", torch.uint8: "u8"}
SCALARS = {"i32": "i", "u32": "I", "i64": "q", "u64": "Q", "fp32": "f", "fp64": "d"}


@dataclass(frozen=True)
class Ref:
    name: str
    offset: int = 0


@dataclass(frozen=True)
class Scalar:
    value: object
    dtype: str = "i32"

    def bits(self):
        if self.dtype not in SCALARS:
            raise UnsupportedKernel(f"Unknown scalar type: {self.dtype}")
        return int.from_bytes(struct.pack("<" + SCALARS[self.dtype], self.value), "little")


class Program:
    def __init__(self, label, dispatch="production"):
        self.label, self.dispatch = label, dispatch
        self.buffers = {}
        self.launches = []

    def buffer(self, name, tensor, *, readonly=False):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in self.buffers:
            raise ValueError(f"Invalid or repeated buffer name: {name}")
        if tensor.device.type != "cpu" or not tensor.is_contiguous() or tensor.dtype not in TYPES:
            raise UnsupportedKernel("Replay requires a contiguous supported CPU tensor")
        self.buffers[name] = {"tensor": tensor.clone(), "readonly": readonly}
        return Ref(name)

    def output(self, name, shape, dtype):
        # Poison output/intermediate storage, including integer routes/counts.
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        tensor = (torch.frombuffer(bytearray(b"\xff" * nbytes), dtype=dtype).reshape(shape).clone()
                  if nbytes else torch.empty(shape, dtype=dtype))
        return self.buffer(name, tensor)

    def launch(self, fn, arguments, grid, *, num_warps=4, num_stages=None):
        if set(arguments) != set(fn.arg_names):
            raise ValueError(f"Arguments differ from {fn.__name__}: {set(arguments) ^ set(fn.arg_names)}")
        grid = tuple(grid) + (1,) * (3 - len(grid))
        if len(grid) != 3 or any(type(x) is not int or x < 1 for x in grid):
            raise ValueError(f"Invalid grid: {grid}")
        if num_warps not in (1, 2, 4, 8, 16, 32):
            raise UnsupportedKernel("Unsupported warp count")
        options = {"num_warps": num_warps}
        if num_stages is not None:
            options["num_stages"] = num_stages
        self.launches.append({"fn": fn, "arguments": arguments, "grid": grid, "options": options})


def compile_harness(folder, library, env):
    exe = folder / "pipeline"
    checked_command(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
                     f"-I{env['CUDA_INSTALL_PATH']}/include", HERE / "pipeline.cpp",
                     f"-L{library.parent}", f"-Wl,-rpath,{library.parent}", "-lcudart", "-ldl", "-o", exe],
                    folder, env, "compile-pipeline.log")
    return exe


def export_launch(program, launch, folder, env, index):
    fn, arguments = launch["fn"], launch["arguments"]
    signature, constants, runtime = {}, {}, []
    for i, name in enumerate(fn.arg_names):
        value = arguments[name]
        if i in fn.constexprs:
            signature[name], constants[name] = "constexpr", value
        elif isinstance(value, Ref):
            if value.name not in program.buffers:
                raise ValueError(f"Unknown buffer: {value.name}")
            tensor = program.buffers[value.name]["tensor"]
            if not 0 <= value.offset <= tensor.numel() * tensor.element_size() or value.offset % tensor.element_size():
                raise ValueError("Unaligned or out-of-range buffer view")
            signature[name] = "*" + TYPES[tensor.dtype]
            runtime.append({"kind": "buffer", "name": value.name, "offset": value.offset})
        elif isinstance(value, Scalar):
            signature[name] = value.dtype
            runtime.append({"kind": "scalar", "bits": value.bits(), "dtype": value.dtype, "value": value.value})
        else:
            raise TypeError(f"Runtime argument {name} requires Ref or Scalar")
    compiled = triton.compile(ASTSource(fn, signature, constexprs=constants),
                              target=GPUTarget("cuda", 90, 32), options=launch["options"])
    ptx, entry = compiled.asm["ptx"], compiled.metadata.name
    prefix = f"kernel{index:03d}"
    (folder / f"{prefix}.ptx").write_text(ptx)
    (folder / f"{prefix}.cubin").write_bytes(compiled.asm["cubin"])
    params = re.search(r"\.entry\s+" + re.escape(entry) + r"\s*\((.*?)\)", ptx, re.S)
    if not params or params.group(1).count(".param") != len(runtime) + 2:
        raise UnsupportedKernel(f"Changed Triton parameter ABI: {entry}")
    if compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
        raise UnsupportedKernel("Nonempty Triton scratch is unsupported")
    target = re.search(r"(?m)^\s*\.target\s+(\w+)", ptx).group(1)
    if target not in ("sm_90", "sm_90a") or getattr(compiled.metadata, "num_ctas", 1) != 1:
        raise UnsupportedKernel("Replay supports a single-CTA SM90 target")
    resources_log, _ = checked_command(["cuobjdump", "--dump-resource-usage", f"{prefix}.cubin"],
                                       folder, env, f"resources{index:03d}.log")
    resources = next(r for r in parse_resource_usage(resources_log.read_text()) if r["function"] == entry)
    record = {"index": index, "entry": entry, "module": f"{prefix}.cubin", "target": target,
              "grid": launch["grid"], "options": launch["options"], "constants": constants,
              "threads": launch["options"]["num_warps"] * 32, "shared_bytes": compiled.metadata.shared,
              "arguments": runtime + [{"kind": "scalar", "bits": 0, "dtype": "u64", "value": 0}] * 2,
              "resources": resources, "ptx_version": re.search(r"\.version\s+([\d.]+)", ptx).group(1),
              "static_instruction_counts": instruction_counts(ptx),
              "ptx_sha256": digest(folder / f"{prefix}.ptx"), "cubin_sha256": digest(folder / f"{prefix}.cubin")}
    record["timing_eligible"] = not (resources.get("stack", 0) or resources.get("local", 0))
    record["timing_status"] = ("UPSTREAM_H100_UNCALIBRATED_FOR_H20" if record["timing_eligible"]
                               else "UNSUPPORTED_CUBIN_SPILLS")
    if not record["timing_eligible"]:
        # Keep actual machine-code evidence. This PTX executor cannot claim its
        # cycle count includes compiler-generated spill instructions.
        sass, _ = checked_command(["cuobjdump", "--dump-sass", f"{prefix}.cubin"],
                                  folder, env, f"sass{index:03d}.log")
        record["sass_stack_or_call_instructions"] = [line.strip() for line in sass.read_text().splitlines()
                                                     if re.search(r"\b(?:LDL|STL|CALL|RET)\b", line)]
    write_json(folder / f"export{index:03d}.json", record)
    (folder / f"{prefix}.ptxinfo").write_text(
        "ptxas info    : 0 bytes gmem\n"
        f"ptxas info    : Compiling entry function '{entry}' for '{target}'\n"
        # Supply only measured register/shared metadata. Do not invent zero
        # spill counts from the cubin resource summary (which omits them).
        f"ptxas info    : Used {resources['reg']} registers, {resources.get('shared', 0)} bytes smem\n")
    return record


def quote(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def export_program(program, folder, env):
    buffers, indices = [], {}
    for name, value in program.buffers.items():
        tensor = value["tensor"]
        raw = raw_tensor(tensor)
        path = folder / f"{name}.input.bin"
        path.write_bytes(GUARD + raw + GUARD)
        indices[name] = len(indices)
        buffers.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "bytes": len(raw), "readonly": value["readonly"], "input_sha256": digest(path)})
    exports = [export_launch(program, launch, folder, env, i) for i, launch in enumerate(program.launches)]
    record = {"protocol": "SIM002_V1", "label": program.label, "dispatch": program.dispatch,
              "guard_bytes_each_side": len(GUARD), "buffers": buffers, "launches": exports}
    write_json(folder / "program.json", record)
    lines = [f"SIM002_V1 {len(buffers)} {len(exports)}"]
    for b in buffers:
        lines.append(f"{quote(b['name'])} {b['bytes']} {int(b['readonly'])} {quote(b['name'] + '.input.bin')}")
    for e in exports:
        lines.append(f"{quote(folder / e['module'])} {quote(e['entry'])} " + " ".join(map(str, e["grid"]))
                     + f" {e['threads']} {e['shared_bytes']} {len(e['arguments'])}")
        for a in e["arguments"]:
            lines.append(f"P {indices[a['name']]} {a['offset']}" if a["kind"] == "buffer" else f"S {a['bits']}")
    (folder / "replay.txt").write_text("\n".join(lines) + "\n")
    return record


def read_snapshot(path, tensor):
    data = path.read_bytes()
    size = tensor.numel() * tensor.element_size()
    if len(data) != size + 2 * len(GUARD):
        raise ValueError(f"Unexpected snapshot size: {path}")
    if data[:len(GUARD)] != GUARD or data[-len(GUARD):] != GUARD:
        raise AssertionError(f"Buffer guard damaged: {path}")
    if not size:
        return torch.empty_like(tensor)
    return torch.frombuffer(bytearray(data[len(GUARD):-len(GUARD)]), dtype=tensor.dtype).reshape(tensor.shape).clone()


def execute(program, folder, source, library, env, executable, timeout=180):
    export = export_program(program, folder, env)
    if not program.launches:
        return {"empty_no_launch": True, "program": export}, {n: v["tensor"].clone() for n, v in program.buffers.items()}
    copy_config(source, folder)
    log, seconds = checked_command(["stdbuf", "-oL", "-eL", executable, folder / "replay.txt", library],
                                   folder, env, "simulation.log", timeout)
    output = log.read_text(errors="replace")
    if f"SIM002_OUTPUTS_WRITTEN {len(program.launches)}" not in output or "this function has not been implemented" in output:
        raise RuntimeError("Incomplete replay or unimplemented CUDA API")
    maps = (folder / "loaded-libraries.txt").read_text()
    if str(library) not in maps or re.search(r"/[^\s]*libcuda\.so", maps):
        raise RuntimeError("Unexpected physical CUDA driver mapping")
    snapshots, final = [], {}
    for i in range(len(program.launches)):
        row = {}
        for name, b in program.buffers.items():
            path = folder / f"step{i:03d}-{name}.bin"
            value = read_snapshot(path, b["tensor"])
            if b["readonly"] and path.read_bytes() != (folder / f"{name}.input.bin").read_bytes():
                raise AssertionError(f"Input changed at step {i}: {name}")
            row[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
            final[name] = value
        snapshots.append(row)
    cycles = [int(x) for x in re.findall(r"(?m)^gpu_tot_sim_cycle\s*=\s*(\d+)", output)]
    eligible = all(e["timing_eligible"] for e in export["launches"])
    return {"program": export, "snapshots": snapshots, "readonly_inputs_unchanged": True,
            "guards_intact_every_step": True, "physical_driver_loaded": False,
            "simulator_wall_seconds": seconds,
            "timing_status": "UPSTREAM_H100_UNCALIBRATED_FOR_H20" if eligible else "UNSUPPORTED_CUBIN_SPILLS",
            "upstream_h100_cycle_counters": cycles if eligible else None,
            "completed_launches": len(program.launches)}, final
