#include "bf16_conversion.h"
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

int main() {
    const uint32_t low_bits[] = {0, 1, 0x7fff, 0x8000, 0x8001, 0xffff};
    unsigned checked = 0;
    for (uint32_t high = 0; high < 65536; ++high) {
        float expanded = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(high)));
        uint32_t expected_expansion;
        std::memcpy(&expected_expansion, &expanded, sizeof(expanded));
        if (sim001::bf16_to_f32_bits(high) != expected_expansion) return 2;
        for (uint32_t low : low_bits) {
            const uint32_t bits = (high << 16) | low;
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            __nv_bfloat16_raw expected = __float2bfloat16_rn(value);
            if (sim001::f32_to_bf16_rn_bits(bits) != expected.x) {
                std::fprintf(stderr, "conversion mismatch: 0x%08x\n", bits);
                return 3;
            }
            ++checked;
        }
    }
    std::printf("BF16_CONVERSION_PASS %u rounding cases; 65536 expansions\n", checked);
    return 0;
}
