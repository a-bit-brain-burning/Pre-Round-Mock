// Bit-exact scalar BF16 conversions for the simulator's finite arithmetic path.
#pragma once
#include <cstdint>

namespace sim001 {
inline uint32_t bf16_to_f32_bits(uint16_t bits) {
    return static_cast<uint32_t>(bits) << 16;
}
inline uint16_t f32_to_bf16_rn_bits(uint32_t bits) {
    // CUDA's canonical BF16 NaN; otherwise round to nearest, ties to even.
    if ((bits & 0x7fffffffU) > 0x7f800000U) return 0x7fffU;
    const uint32_t bias = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>((bits + bias) >> 16);
}
}  // namespace sim001
