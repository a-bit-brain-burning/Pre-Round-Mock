"""Apply version-checked conversion, fused FMA and predicated atomic fixes."""

import json
import subprocess
from pathlib import Path

from bootstrap import HERE, LOCK, digest, timestamp
import patch_wgmma

INSERT = """
  // sim001: scalar BF16 is outside the legacy 11x11 conversion table.
  if (to_type == BF16_TYPE || from_type == BF16_TYPE) {
    if (saturation_mode || pI->is_neg() || pI->get_num_operands() != 2 ||
        (rounding_mode != RN_OPTION && rounding_mode != RNI_OPTION)) {
      fprintf(stderr, "Unsupported BF16 cvt modifiers: %s\\n", pI->to_string().c_str());
      abort();
    }
    ptx_reg_t value = thread->get_operand_value(src1, dst, from_type, thread, 1);
    ptx_reg_t result;
    result.u64 = 0;
    if (from_type == BF16_TYPE && to_type == F32_TYPE) {
      result.u32 = sim001::bf16_to_f32_bits(value.u16);
    } else if (from_type == F32_TYPE && to_type == BF16_TYPE) {
      result.u16 = sim001::f32_to_bf16_rn_bits(value.u32);
    } else {
      fprintf(stderr, "Unsupported BF16 cvt types: %s\\n", pI->to_string().c_str());
      abort();
    }
    thread->set_operand_value(dst, result, to_type, thread, pI);
    return;
  }

"""
GUARD = """  if (src_fmt >= sizeof(g_cvt_fn) / sizeof(g_cvt_fn[0]) ||
      dst_fmt >= sizeof(g_cvt_fn[0]) / sizeof(g_cvt_fn[0][0])) {
    fprintf(stderr, "Unsupported cvt type pair: %s\\n", pI->to_string().c_str());
    abort();
  }

"""


FMA = r'''void fma_impl(const ptx_instruction *pI, ptx_thread_info *thread) {
  // sim002: PTX fma rounds once; a host a*b+c expression may round twice.
  const unsigned type = pI->get_type();
  if ((type != F32_TYPE && type != F64_TYPE && type != FF64_TYPE) ||
      pI->get_scalar_type().size() != 1 || pI->saturation_mode() ||
      pI->to_string().find(".ftz") != std::string::npos) {
    fprintf(stderr, "Unsupported sim002 FMA variant: %s\n", pI->to_string().c_str());
    abort();
  }
  const operand_info &dst = pI->dst();
  const ptx_reg_t a = thread->get_operand_value(pI->src1(), dst, type, thread, 1);
  const ptx_reg_t b = thread->get_operand_value(pI->src2(), dst, type, thread, 1);
  const ptx_reg_t c = thread->get_operand_value(pI->src3(), dst, type, thread, 1);
  int mode;
  switch (pI->rounding_mode()) {
    case RN_OPTION: mode = FE_TONEAREST; break;
    case RZ_OPTION: mode = FE_TOWARDZERO; break;
    case RM_OPTION: mode = FE_DOWNWARD; break;
    case RP_OPTION: mode = FE_UPWARD; break;
    default:
      fprintf(stderr, "Unsupported FMA rounding mode\n");
      abort();
  }
  const int previous = fegetround();
  if (fesetround(mode)) abort();
  ptx_reg_t result;
  result.u64 = 0;
  if (type == F32_TYPE) result.f32 = std::fma(a.f32, b.f32, c.f32);
  else result.f64 = std::fma(a.f64, b.f64, c.f64);
  if (fesetround(previous)) abort();
  thread->set_operand_value(dst, result, type, thread, pI);
}'''


def p1_source(source):
    relative = "src/cuda-sim/instructions.cc"
    original = subprocess.check_output(["git", "show", f"{LOCK['revision']}:{relative}"],
                                       cwd=source, text=True)
    anchor = "  unsigned saturation_mode = pI->saturation_mode();\n\n\n  int to_sign, from_sign;"
    table = "  if (g_cvt_fn[src_fmt][dst_fmt] != NULL) {"
    if original.count(anchor) != 1 or original.count(table) != 1:
        raise RuntimeError("Pinned backend patch anchors changed")
    revised = original.replace('#include "instructions.h"',
                               '#include "instructions.h"\n#include "sim001_bf16_conversion.h"', 1)
    revised = revised.replace(anchor, "  unsigned saturation_mode = pI->saturation_mode();\n"
                              + INSERT + "  int to_sign, from_sign;", 1)
    revised = revised.replace(table, GUARD + table, 1)
    return relative, original, revised


def fma_source(source):
    relative, original, p1 = p1_source(source)
    anchor = "void fma_impl(const ptx_instruction *pI, ptx_thread_info *thread) {\n  mad_def(pI, thread);\n}"
    if p1.count(anchor) != 1:
        raise RuntimeError("Pinned FMA implementation changed")
    revised = p1.replace(anchor, FMA).replace('#include "instructions.h"', '#include "instructions.h"\n#include <cmath>', 1)
    return relative, original, revised


PACKED = r'''
  // sim002: explicitly implement only the packed conversions used by the kernels.
  if (to_type == BF16X2_TYPE || from_type == BF16X2_TYPE ||
      to_type == E4M3X2_TYPE || from_type == E4M3X2_TYPE) {
    ptx_reg_t result;
    result.u64 = 0;
    if (saturation_mode || pI->is_neg() ||
        (rounding_mode != RN_OPTION && rounding_mode != RNI_OPTION)) {
      fprintf(stderr, "Unsupported packed cvt modifiers: %s\n", pI->to_string().c_str());
      abort();
    }
    const ptx_reg_t a = thread->get_operand_value(src1, dst, from_type, thread, 1);
    if (to_type == BF16X2_TYPE && from_type == F32_TYPE && pI->get_num_operands() == 3) {
      const ptx_reg_t b = thread->get_operand_value(pI->src2(), dst, from_type, thread, 1);
      result.u32 = sim002::f32_pair_to_bf16x2(a.u32, b.u32);
    } else if (to_type == F16X2_TYPE && from_type == E4M3X2_TYPE && pI->get_num_operands() == 2) {
      result.u32 = sim002::e4m3x2_to_f16x2(a.u16);
    } else {
      fprintf(stderr, "Unsupported packed cvt types: %s\n", pI->to_string().c_str());
      abort();
    }
    thread->set_operand_value(dst, result, to_type, thread, pI);
    return;
  }
'''


def packed_source(source):
    relative, original, revised = fma_source(source)
    revised = revised.replace('#include "sim001_bf16_conversion.h"',
                              '#include "sim001_bf16_conversion.h"\n#include "sim002_numeric_conversions.h"')
    revised = revised.replace(INSERT, PACKED + INSERT)
    anchor = "      result.u16 = sim001::f32_to_bf16_rn_bits(value.u32);\n    } else {"
    replacement = """      result.u16 = sim001::f32_to_bf16_rn_bits(value.u32);
    } else if (from_type == BF16_TYPE && to_type == F64_TYPE) {
      result.f64 = sim002::bf16_to_f64(value.u16);
    } else if (from_type == F64_TYPE && to_type == BF16_TYPE) {
      result.u16 = sim002::f64_to_bf16_rn_bits(value.u64);
    } else {"""
    if revised.count(anchor) != 1:
        raise RuntimeError("BF16 extension anchor changed")
    return relative, original, revised.replace(anchor, replacement)


def p2_source(source):
    relative, original, revised = packed_source(source)
    anchor = "      result.u16 = sim002::f64_to_bf16_rn_bits(value.u64);\n    } else {"
    replacement = """      result.u16 = sim002::f64_to_bf16_rn_bits(value.u64);
    } else if (from_type == F16_TYPE && to_type == BF16_TYPE) {
      result.u16 = sim002::f16_to_bf16(value.u16);
    } else if (from_type == BF16_TYPE && to_type == F16_TYPE) {
      result.u16 = sim002::bf16_to_f16(value.u16);
    } else {"""
    if revised.count(anchor) != 1:
        raise RuntimeError("FP16/BF16 conversion anchor changed")
    return relative, original, revised.replace(anchor, replacement)


def expected_source(source):
    relative, original, revised = p2_source(source)
    revised = revised.replace('#include "sim002_numeric_conversions.h"',
        '#include "sim002_numeric_conversions.h"\n#include "../gpgpu-sim/flash/wgmma/sim003_wgmma_numeric.h"', 1)
    anchor = "  // sim002: explicitly implement only the packed conversions used by the kernels."
    addition = r'''
  // sim003: Router prepacking requires ties-to-even, including subnormals.
  // The legacy half library rounds exact ties away from zero.
  if (from_type == F32_TYPE && (to_type == F16_TYPE || to_type == F16X2_TYPE) &&
      (rounding_mode == RN_OPTION || rounding_mode == RNI_OPTION)) {
    const bool packed = to_type == F16X2_TYPE;
    if (saturation_mode || pI->is_neg() || pI->get_num_operands() != (packed ? 3 : 2)) {
      fprintf(stderr, "Unsupported sim003 FP16 cvt modifiers: %s\n", pI->to_string().c_str());
      abort();
    }
    const ptx_reg_t a = thread->get_operand_value(src1, dst, from_type, thread, 1);
    ptx_reg_t result;
    result.u64 = 0;
    result.u16 = sim003::f32_to_f16_rn(a.f32);
    if (packed) {
      const ptx_reg_t b = thread->get_operand_value(pI->src2(), dst, from_type, thread, 1);
      result.u32 = (uint32_t(result.u16) << 16) | sim003::f32_to_f16_rn(b.f32);
    }
    thread->set_operand_value(dst, result, to_type, thread, pI);
    return;
  }
'''
    if revised.count(anchor) != 1:
        raise RuntimeError("FP16 conversion patch anchor changed")
    return relative, original, revised.replace(anchor, addition + anchor, 1)


def original_file(source, relative):
    return subprocess.check_output(["git", "show", f"{LOCK['revision']}:{relative}"], cwd=source, text=True)


def parser_sources(source):
    result = {}
    for relative in ("src/cuda-sim/ptx.l", "src/cuda-sim/ptx.y", "src/cuda-sim/ptx_ir.cc",
                     "src/gpgpu-sim/flash/ptx_sched/ptx_scheduler.cc"):
        original = original_file(source, relative)
        if relative.endswith("ptx.l"):
            anchor = r"\.bf16  TC;  return BF16_TYPE;"
            extra = r"\.bf16x2  TC;  return BF16X2_TYPE;" + "\n" + r"\.e4m3x2  TC;  return E4M3X2_TYPE;"
            revised = original.replace(anchor, anchor + "\n" + extra)
        elif relative.endswith("ptx.y"):
            revised = original.replace("%token  BF16_TYPE", "%token  BF16_TYPE\n%token  BF16X2_TYPE\n%token  E4M3X2_TYPE")
            anchor = "\t| BF16_TYPE  { recognizer->add_scalar_type_spec( BF16_TYPE ); }"
            extra = "\n\t| BF16X2_TYPE { recognizer->add_scalar_type_spec( BF16X2_TYPE ); }\n\t| E4M3X2_TYPE { recognizer->add_scalar_type_spec( E4M3X2_TYPE ); }"
            revised = revised.replace(anchor, anchor + extra)
        elif relative.endswith("ptx_ir.cc"):
            anchor = "    case F16X2_TYPE:\n      size = 32;"
            revised = original.replace(anchor, "    case BF16X2_TYPE:\n      size = 32; basic_type = -1; return 24;\n    case E4M3X2_TYPE:\n      size = 16; basic_type = -1; return 25;\n" + anchor)
        else:
            anchor = "    case BF16_TYPE:"
            revised = original.replace(anchor, anchor + "\n    case BF16X2_TYPE:\n    case E4M3X2_TYPE:")
        if revised == original:
            raise RuntimeError(f"Packed type patch anchor changed: {relative}")
        result[relative] = (original, revised)
    return result


def atomic_sources(source):
    relative = "src/gpgpu-sim/shader.cc"
    original = original_file(source, relative)
    anchor = "  if (inst.isatomic()) m_warp[inst.warp_id()]->inc_n_atomic();"
    replacement = """  // sim002: ptx_exec_inst clears the lane mask when its predicate is false.
  // Only issued lanes get memory responses that decrement this counter.
  if (inst.isatomic() && inst.active(t))
    m_warp[inst.warp_id()]->inc_n_atomic();"""
    if original.count(anchor) != 1:
        raise RuntimeError("Predicated atomic counter anchor changed")
    return {relative: (original, original.replace(anchor, replacement))}


def helper_sources():
    return {**patch_wgmma.helpers(), "src/cuda-sim/sim001_bf16_conversion.h": (HERE / "bf16_conversion.h").read_bytes(),
            "src/cuda-sim/sim002_numeric_conversions.h": (HERE / "numeric_conversions.h").read_bytes().replace(
                b'#include "bf16_conversion.h"', b'#include "sim001_bf16_conversion.h"')}


def verify(source):
    relative, _, revised = expected_source(source)
    target = source / relative
    expected = {relative: revised, **{p: pair[1] for p, pair in
                {**parser_sources(source), **atomic_sources(source), **patch_wgmma.sources(source, original_file)}.items()}}
    for p, text in expected.items():
        if (source / p).read_text() != text:
            raise RuntimeError(f"Backend source differs from reviewed implementation: {p}; rebuild first")
    for p, data in helper_sources().items():
        if (source / p).read_bytes() != data:
            raise RuntimeError(f"Backend helper differs: {p}")
    return {p: digest(source / p) for p in [*expected, *helper_sources()]}


def apply(source, root):
    relative, original, revised = expected_source(source)
    target = source / relative
    if target.read_text() not in (original, p1_source(source)[2], fma_source(source)[2], packed_source(source)[2], p2_source(source)[2], revised):
        raise RuntimeError("Refusing to overwrite unrelated upstream changes")
    parsers = {**parser_sources(source), **atomic_sources(source), **patch_wgmma.sources(source, original_file)}
    for p, (before, after) in parsers.items():
        # Preserve the reviewed first P3 build while accepting its narrow fix.
        prior = {"src/gpgpu-sim/flash/wgmma/wgmma_f16.cc":
                 "a2281d4fe6e46160d019d73d54a0609a20cfed7652419af38dc374bf6aa8fa93"}
        if (source / p).read_text() not in (before, after) and digest(source / p) != prior.get(p):
            raise RuntimeError(f"Refusing unrelated parser change: {p}")
    for p, data in helper_sources().items():
        if (source / p).exists() and (source / p).read_bytes() != data:
            if p != "src/cuda-sim/sim002_numeric_conversions.h" or digest(source / p) != "422d0d674913d27d9d400d01f0c9b102875092c83e8e6a8e2cf9d4625a2cf56e":
                raise RuntimeError(f"Refusing unrelated helper change: {p}")
    if target.read_text() != revised:
        target.write_text(revised)
    for p, (_, after) in parsers.items():
        if (source / p).read_text() != after:
            (source / p).write_text(after)
    for p, data in helper_sources().items():
        if not (source / p).exists():
            (source / p).write_bytes(data)
        elif (source / p).read_bytes() != data:
            with (root / "logs" / f"backend-helper-prior-{timestamp()}.h").open("xb") as stream:
                stream.write((source / p).read_bytes())
            (source / p).write_bytes(data)
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=source, text=True)
    output = root / "logs" / f"backend-patch-{timestamp()}.diff"
    with output.open("x") as stream:
        stream.write(diff)
    record = {"revision": LOCK["revision"], "diff": str(output), "diff_sha256": digest(output),
              "source_sha256": verify(source)}
    print(json.dumps({"backend_patch": record}), flush=True)
    return record
