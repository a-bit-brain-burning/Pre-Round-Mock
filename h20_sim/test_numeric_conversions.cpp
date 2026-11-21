#include "numeric_conversions.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdio>
#include <limits>

static uint64_t double_bits(double value) {
    uint64_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
static bool check_double(double value) {
    __nv_bfloat16_raw expected = __double2bfloat16(value);
    uint16_t actual = sim002::f64_to_bf16_rn_bits(double_bits(value));
    if (actual != expected.x) {
        std::fprintf(stderr, "FP64 conversion mismatch %.17g: %04x != %04x\n", value, actual, expected.x);
        return false;
    }
    return true;
}
int main() {
    unsigned midpoint_tests = 0;
    for (unsigned i = 0; i < 0x7f7f; ++i) {
        const double lo = sim002::bf16_to_f64(i), hi = sim002::bf16_to_f64(i + 1);
        const double mid = (lo + hi) / 2;
        for (double x : {std::nextafter(mid, lo), mid, std::nextafter(mid, hi)})
            for (double sign : {1., -1.}) {
                if (!check_double(sign * x)) return 2;
                ++midpoint_tests;
            }
    }
    for (double x : {0., -0., std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(),
                      std::numeric_limits<double>::max(), std::numeric_limits<double>::denorm_min()})
        if (!check_double(x)) return 3;
    for (unsigned bits = 0; bits < 65536; ++bits) {
        const __nv_bfloat16_raw bf = __float2bfloat16_rn(__half2float(__ushort_as_half(bits)));
        const __half_raw hf = __float2half_rn(__bfloat162float(__ushort_as_bfloat16(bits)));
        if (sim002::f16_to_bf16(bits) != bf.x || sim002::bf16_to_f16(bits) != hf.x) {
            std::fprintf(stderr, "FP16/BF16 conversion mismatch: %04x\n", bits);
            return 6;
        }
        const __half2_raw expected = __nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3);
        const uint32_t packed = uint32_t(expected.x) | (uint32_t(expected.y) << 16);
        if (sim002::e4m3x2_to_f16x2(bits) != packed) {
            std::fprintf(stderr, "FP8 conversion mismatch: %04x\n", bits);
            return 4;
        }
        const uint32_t a_bits = (bits << 16) | 0x8001, b_bits = (~bits << 16) | 0x7fff;
        float a, b;
        std::memcpy(&a, &a_bits, 4); std::memcpy(&b, &b_bits, 4);
        const __nv_bfloat162_raw pair = __floats2bfloat162_rn(b, a);
        if (sim002::f32_pair_to_bf16x2(a_bits, b_bits) != (uint32_t(pair.x) | (uint32_t(pair.y) << 16)))
            return 5;
    }
    std::printf("NUMERIC_CONVERSIONS_PASS %u FP64 midpoint samples; 65536 FP8 pairs; 65536 BF16 pairs; 65536 FP16/BF16 both directions\n", midpoint_tests);
}
