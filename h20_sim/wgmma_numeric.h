// sim003 model choice: FP32 accumulation, then round FP16 output to nearest-even.
// PTX leaves matrix accumulation order/rounding unspecified; this is not H20 data.
#pragma once
#include <cstdint>
#include <cstring>

namespace sim003 {
inline uint32_t round_shift(uint32_t value, unsigned shift) {
    const uint32_t q = value >> shift;
    const uint32_t rest = value & ((uint32_t(1) << shift) - 1);
    const uint32_t half = uint32_t(1) << (shift - 1);
    return q + (rest > half || (rest == half && (q & 1)));
}
inline uint16_t f32_to_f16_rn(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = (bits >> 16) & 0x8000;
    const unsigned exponent = (bits >> 23) & 255;
    const uint32_t fraction = bits & 0x7fffff;
    if (exponent == 255) return fraction ? 0x7fff : sign | 0x7c00;
    if (exponent > 142) return sign | 0x7c00;
    if (exponent < 102) return sign;
    if (exponent <= 112) return sign | round_shift(0x800000 | fraction, 126 - exponent);
    return sign | (((exponent - 112) << 10) + round_shift(fraction, 13));
}
}
