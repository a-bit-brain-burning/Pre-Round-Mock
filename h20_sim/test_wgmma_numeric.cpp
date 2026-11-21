#include "wgmma_numeric.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>

static bool check(float x) {
    const __half_raw expected = __float2half_rn(x);
    const uint16_t actual = sim003::f32_to_f16_rn(x);
    if (actual == expected.x) return true;
    std::fprintf(stderr, "FP16 rounding mismatch: %.9g %04x != %04x\n", x, actual, expected.x);
    return false;
}
int main() {
    unsigned count = 0;
    for (unsigned i = 0; i < 0x7bff; ++i) {
        const float a = __half2float(__ushort_as_half(i));
        const float b = __half2float(__ushort_as_half(i + 1));
        const float mid = (a + b) / 2;
        for (float x : {std::nextafter(mid, a), mid, std::nextafter(mid, b)})
            for (float sign : {1.f, -1.f}) {
                if (!check(sign * x)) return 2;
                ++count;
            }
    }
    uint32_t state = 1108;
    for (unsigned i = 0; i < 65536; ++i) {
        state = state * 1664525u + 1013904223u;
        float x;
        std::memcpy(&x, &state, 4);
        if (!check(x)) return 3;
        ++count;
    }
    for (float x : {0.f, -0.f, 65504.f, 65520.f, -65520.f,
                   std::nextafter(65520.f, 0.f), std::nextafter(65520.f, 1e6f),
                   std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                   std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::denorm_min()}) {
        if (!check(x)) return 4;
        ++count;
    }
    std::printf("WGMMA_NUMERIC_PASS %u cases\n", count);
}
