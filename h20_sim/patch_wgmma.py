"""Version-checked support for the two missing Router m64n64k16 variants."""

from bootstrap import HERE


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError(f"Pinned WGMMA patch anchor changed: {before[:80]}")
    return text.replace(before, after, 1)


def sources(source, original_file):
    base = "src/gpgpu-sim/flash/wgmma/"
    result = {}
    path = base + "tensor_wgmma.cc"
    original = original_file(source, path)
    text = replace_once(original,
        "  if (accumulator_type == F32_TYPE && a_type == F16_TYPE &&\n",
        """  // sim003: Router needs N=64 BF16/F32 and FP16/FP16 accumulators.
  // Preserve existing F16/F32 widths; other new widths remain unsupported.
  const bool router_variant = pI->get_wgmma_shape_n() == 64 &&
      ((accumulator_type == F32_TYPE && a_type == BF16_TYPE && b_type == BF16_TYPE) ||
       (accumulator_type == F16_TYPE && a_type == F16_TYPE && b_type == F16_TYPE));
  if (router_variant && pI->get_wgmma_shape_k() == 16) {
    wgmma_m64nXk16_f16_impl(pI, core, inst);
    return;
  }

  if (accumulator_type == F32_TYPE && a_type == F16_TYPE &&
""")
    text = replace_once(text, "      scalar_type_bytes(accumulator_type);",
                        "      (accumulator_type == F16_TYPE ? 4 : scalar_type_bytes(accumulator_type));")
    result[path] = (original, text)
    path = base + "wgmma_f16.cc"
    original = original_file(source, path)
    text = replace_once(original, '#include "tensor_wgmma.h"',
                        '#include "tensor_wgmma.h"\n#include "sim003_wgmma_numeric.h"\n#include <cstdio>\n#include <cstdlib>')
    text = replace_once(text, "  int N = pI->get_wgmma_shape_n();", """  int N = pI->get_wgmma_shape_n();
  const std::list<int> &types = pI->get_scalar_type();
  auto type = types.begin();
  const bool half_accumulator = *type++ == F16_TYPE;
  const bool bf16_input = *type == BF16_TYPE;
  const unsigned elements_per_register = half_accumulator ? 2 : 1;
  const auto decode = [bf16_input](uint16_t bits) {
    return bf16_input ? mma_bf16_to_f32(bits) : mma_f16_to_f32(bits);
  };
  const auto unsupported = [pI](const char *reason) {
    fprintf(stderr, "Unsupported sim003 WGMMA %s: %s\\n", reason, pI->to_string().c_str());
    abort();
  };""")
    text = replace_once(text, """      dst.get_vect_nelem() != static_cast<unsigned>(N / 2)) {
    return;""", """      dst.get_vect_nelem() != static_cast<unsigned>(N / (2 * elements_per_register))) {
    unsupported("accumulator fragment");""")
    text = replace_once(text, "  ptx_reg_t b_desc_reg = representative_thread->get_operand_value(", """  if (thread_count != 128 || operand_count != (a_from_register ? 7 : 8) ||
      (scale_a != 1 && scale_a != -1) || (scale_b != 1 && scale_b != -1) ||
      major_a < 0 || major_a > 1 || major_b < 0 || major_b > 1)
    unsupported("operands or warpgroup");

  ptx_reg_t b_desc_reg = representative_thread->get_operand_value(""")
    # Do not silently map invalid scale operands to +1.
    text = text.replace("scale.s32 == -1 ? -1 : 1", "scale.s32")
    text = text.replace("mma_f16_to_f32(b_bits)", "decode(b_bits)")
    text = text.replace("mma_f16_to_f32(a_bits)", "decode(a_bits)")
    text = text.replace("mma_f16_to_f32(packed_f16(a_regs[reg],", "decode(packed_f16(a_regs[reg],")
    text = replace_once(text, "    if (src_a.get_vect_nelem() != 4)\n      return;",
                        '    if (src_a.get_vect_nelem() != 4)\n      unsupported("A fragment");')
    text = replace_once(text, "    bool include_c = (pred.pred & 0x1u) == 0;", """    // Predicate registers use inverted internal encoding, integer literals
    // do not. Router emits literal 0 to reset each residual BK accumulator.
    if (scale_d_pred.is_literal() && pred.u32 > 1)
      unsupported("scale-d literal");
    bool include_c = scale_d_pred.is_literal() ? pred.u32 != 0
                                              : (pred.pred & 0x1u) == 0;""")
    text = replace_once(text, "      ptx_reg_t c_reg = thread->get_reg(dst.vec_symbol(reg));", """      ptx_reg_t c_reg;
      c_reg.u64 = 0;
      if (include_c) c_reg = thread->get_reg(dst.vec_symbol(reg));""")
    start = text.index("      int row = 0;", text.index("unsigned dst_elements"))
    end = text.index("      thread->set_reg(dst.vec_symbol(reg), d_reg);", start)
    text = text[:start] + """      ptx_reg_t d_reg;
      d_reg.u64 = 0;
      for (unsigned half = 0; half < elements_per_register; ++half) {
        const unsigned element = reg * elements_per_register + half;
        int row = 0, col_in_tile = 0;
        wgmma_m64n8_accumulator_coord(lane, element % 4, row, col_in_tile);
        const int col = static_cast<int>((element / 4) * 8) + col_in_tile;
        // Model choice: ascending K, FP32 sum, then add C. PTX does not
        // specify the hardware reduction tree or intermediate rounding.
        float product = 0.0f;
        for (int k = 0; k < K; ++k)
          product += A_mat[row * K + k] * B_mat[col * K + k];
        const float c = !include_c ? 0.0f : half_accumulator
            ? mma_f16_to_f32(packed_f16(c_reg, half)) : c_reg.f32;
        const float value = product + c;
        if (half_accumulator)
          d_reg.u32 |= uint32_t(sim003::f32_to_f16_rn(value)) << (16 * half);
        else
          d_reg.f32 = value;
      }
""" + text[end:]
    result[path] = (original, text)
    return result


def helpers():
    return {"src/gpgpu-sim/flash/wgmma/sim003_wgmma_numeric.h": (HERE / "wgmma_numeric.h").read_bytes()}
