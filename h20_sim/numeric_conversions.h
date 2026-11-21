// Scalar FP64/BF16 and packed E4M3 conversions used by sim002.
#pragma once
#include "bf16_conversion.h"
#include <cstdint>
#include <cstring>

namespace sim002 {
inline uint64_t round_shift_even(uint64_t value, unsigned shift) {
    const uint64_t quotient = value >> shift;
    const uint64_t remainder = value & ((uint64_t(1) << shift) - 1);
    const uint64_t half = uint64_t(1) << (shift - 1);
    return quotient + (remainder > half || (remainder == half && (quotient & 1)));
}
inline uint16_t f64_to_bf16_rn_bits(uint64_t bits) {
    const uint16_t sign = (bits >> 48) & 0x8000;
    const unsigned raw_exp = (bits >> 52) & 0x7ff;
    const uint64_t fraction = bits & ((uint64_t(1) << 52) - 1);
    if (raw_exp == 0x7ff) return fraction ? 0x7fff : sign | 0x7f80;
    const int exponent = static_cast<int>(raw_exp) - 1023;
    if (exponent < -134) return sign;
    if (exponent > 127) return sign | 0x7f80;
    const uint64_t significand = (uint64_t(1) << 52) | fraction;
    if (exponent < -126) return sign | round_shift_even(significand, -exponent - 81);
    uint64_t rounded = round_shift_even(significand, 45);
    int target_exp = exponent + 127;
    if (rounded == 256) { rounded = 128; ++target_exp; }
    if (target_exp >= 255) return sign | 0x7f80;
    return sign | (target_exp << 7) | (rounded - 128);
}
inline double bf16_to_f64(uint16_t bits) {
    uint32_t expanded = sim001::bf16_to_f32_bits(bits);
    float value;
    std::memcpy(&value, &expanded, sizeof(value));
    return static_cast<double>(value);
}
inline uint16_t e4m3_to_f16(uint8_t bits) {
    const uint16_t sign = (bits & 0x80) << 8;
    unsigned exponent = (bits >> 3) & 15, fraction = bits & 7;
    if (exponent == 15 && fraction == 7) return 0x7fff;
    if (exponent) return sign | ((exponent + 8) << 10) | (fraction << 7);
    if (!fraction) return sign;
    unsigned shifts = 0;
    while (fraction < 8) { fraction <<= 1; ++shifts; }
    return sign | ((9 - shifts) << 10) | ((fraction - 8) << 7);
}
inline uint32_t e4m3x2_to_f16x2(uint16_t bits) {
    return uint32_t(e4m3_to_f16(bits & 0xff)) | (uint32_t(e4m3_to_f16(bits >> 8)) << 16);
}
inline uint32_t f32_pair_to_bf16x2(uint32_t upper, uint32_t lower) {
    return (uint32_t(sim001::f32_to_bf16_rn_bits(upper)) << 16) | sim001::f32_to_bf16_rn_bits(lower);
}
inline uint16_t f16_to_bf16(uint16_t bits) {
    const uint32_t sign = uint32_t(bits & 0x8000) << 16;
    unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    uint32_t single;
    if (exponent == 31) single = sign | 0x7f800000 | (fraction << 13);
    else if (exponent) single = sign | ((exponent + 112) << 23) | (fraction << 13);
    else if (!fraction) single = sign;
    else {
        unsigned shifts = 0;
        while (fraction < 1024) { fraction <<= 1; ++shifts; }
        single = sign | ((113 - shifts) << 23) | ((fraction - 1024) << 13);
    }
    return sim001::f32_to_bf16_rn_bits(single);
}
inline uint16_t bf16_to_f16(uint16_t bits) {
    const uint16_t sign = bits & 0x8000;
    const unsigned raw_exp = (bits >> 7) & 255, fraction = bits & 127;
    if (raw_exp == 255) return fraction ? 0x7fff : sign | 0x7c00;
    const int exponent = static_cast<int>(raw_exp) - 127;
    if (exponent > 15) return sign | 0x7c00;
    if (exponent < -25) return sign;
    if (exponent >= -14) return sign | ((exponent + 15) << 10) | (fraction << 3);
    const uint64_t significand = 128 + fraction;
    return sign | (exponent >= -17 ? significand << (exponent + 17)
                                  : round_shift_even(significand, -exponent - 17));
}
}
