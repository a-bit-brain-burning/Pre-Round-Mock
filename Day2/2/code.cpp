#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int threads = 256;
constexpr int elements = threads * 8;

__device__ double2 reduce_pair(double2 value) {
    __shared__ double2 warp_totals[threads / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int step = 16; step > 0; step /= 2) {
        value.x += __shfl_down_sync(0xffffffff, value.x, step);
        value.y += __shfl_down_sync(0xffffffff, value.y, step);
    }
    if (lane == 0) warp_totals[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < threads / 32 ? warp_totals[lane] : make_double2(0, 0);
        for (int step = 4; step > 0; step /= 2) {
            value.x += __shfl_down_sync(0xffffffff, value.x, step);
            value.y += __shfl_down_sync(0xffffffff, value.y, step);
        }
    }
    return value;
}

__global__ void partial_moments(const float* input, double2* partials,
                                size_t batches, size_t features, size_t spatial, unsigned tiles) {
    const unsigned tile = blockIdx.x % tiles;
    const size_t row = blockIdx.x / tiles;
    const size_t feature = row % features;
    const double origin = input[feature * spatial];
    const float* plane = input + row * spatial;
    double2 value = make_double2(0, 0);
    #pragma unroll
    for (int i = 0; i < elements; i += threads) {
        const size_t pos = size_t(tile) * elements + threadIdx.x + i;
        if (pos < spatial) {
            const double delta = double(plane[pos]) - origin;
            value.x += delta;
            value.y = fma(delta, delta, value.y);
        }
    }
    value = reduce_pair(value);
    if (threadIdx.x == 0) partials[(feature * batches + row / features) * tiles + tile] = value;
}

__global__ void channel_stats(const float* input, const double2* partials, double2* stats,
                              size_t batches, size_t spatial, unsigned tiles) {
    const size_t feature = blockIdx.x;
    const size_t count = batches * tiles;
    double2 value = make_double2(0, 0);
    for (size_t i = threadIdx.x; i < count; i += threads) {
        const double2 part = partials[feature * count + i];
        value.x += part.x;
        value.y += part.y;
    }
    value = reduce_pair(value);
    if (threadIdx.x == 0) {
        const double scale = 1.0 / double(batches * spatial);
        const double centered_mean = value.x * scale;
        const double variance = fmax(0.0, fma(-centered_mean, centered_mean, value.y * scale));
        stats[feature] = make_double2(double(input[feature * spatial]) + centered_mean,
                                      rsqrt(variance + 1.0e-5));
    }
}

__global__ void normalize(const float* input, float* output, const double2* stats,
                          size_t features, size_t spatial, unsigned tiles) {
    const unsigned tile = blockIdx.x % tiles;
    const size_t row = blockIdx.x / tiles;
    const double2 stat = stats[row % features];
    #pragma unroll
    for (int i = 0; i < elements; i += threads) {
        const size_t pos = size_t(tile) * elements + threadIdx.x + i;
        if (pos < spatial) {
            const size_t index = row * spatial + pos;
            output[index] = float((double(input[index]) - stat.x) * stat.y);
        }
    }
}

constexpr int packed_elements = 4096;

template<int packs>
__device__ __forceinline__ double2 moments(const float4 (&values)[packs], double origin) {
    // 每个线程选用自身的首个元素计算差值，再用 FP64 将部分和变换到通道的共同原点。
    const float center = values[0].x;
    float sum = 0, square = 0;
    #pragma unroll
    for (int i = 0; i < packs; ++i) {
        const float4 v = values[i];
        const float a = v.x - center, b = v.y - center, c = v.z - center, d = v.w - center;
        sum += (a + b) + (c + d);
        square += fmaf(a, a, b * b) + fmaf(c, c, d * d);
    }
    if (isfinite(square)) {
        const double shift = double(center) - origin;
        const double shifted_sum = (4 * packs) * shift;
        return make_double2(shifted_sum + sum, fma(shifted_sum, shift, fma(2 * shift, double(sum), double(square))));
    }
    double2 wide = make_double2(0, 0);
    #pragma unroll
    for (int i = 0; i < packs; ++i) {
        const double a = double(values[i].x) - origin, b = double(values[i].y) - origin;
        const double c = double(values[i].z) - origin, d = double(values[i].w) - origin;
        wide.x += (a + b) + (c + d);
        wide.y += (a * a + b * b) + (c * c + d * d);
    }
    return wide;
}

__device__ __forceinline__ float4 apply_stats(float4 v, double2 stat) {
    if (fabs(stat.x) > 1e30 || stat.y < 1e-30) {
        return make_float4((double(v.x) - stat.x) * stat.y, (double(v.y) - stat.x) * stat.y,
                           (double(v.z) - stat.x) * stat.y, (double(v.w) - stat.x) * stat.y);
    }
    const float center = float(stat.x);
    const float residual = float(stat.x - double(center));
    const float scale = float(stat.y);
    return make_float4(((v.x - center) - residual) * scale, ((v.y - center) - residual) * scale,
                       ((v.z - center) - residual) * scale, ((v.w - center) - residual) * scale);
}

__global__ void packed_partials(const float4* input, double2* partials,
                                size_t batches, size_t features, size_t spatial4) {
    const size_t row = size_t(blockIdx.z) * features + blockIdx.y;
    const size_t first = row * spatial4 + size_t(blockIdx.x) * (packed_elements / 4) + threadIdx.x;
    float4 values[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) values[i] = input[first + i * threads];
    const double origin = input[size_t(blockIdx.y) * spatial4].x;
    const double2 total = reduce_pair(moments(values, origin));
    if (threadIdx.x == 0) {
        partials[(size_t(blockIdx.y) * batches + blockIdx.z) * gridDim.x + blockIdx.x] = total;
    }
}

__global__ void packed_output(const float4* input, float4* output, const double2* stats,
                              size_t features, size_t spatial4) {
    const size_t row = size_t(blockIdx.z) * features + blockIdx.y;
    const size_t first = row * spatial4 + size_t(blockIdx.x) * (packed_elements / 4) + threadIdx.x;
    const double2 stat = stats[blockIdx.y];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const size_t pos = first + i * threads;
        output[pos] = apply_stats(input[pos], stat);
    }
}

__global__ void small_channel(const float4* input, float4* output) {
    constexpr int spatial4 = 4096 / 4;
    constexpr int packs = 8 * spatial4 / threads;
    const int feature = blockIdx.x;
    const double origin = input[feature * spatial4].x;
    float4 values[packs];
    #pragma unroll
    for (int i = 0; i < packs; ++i) {
        const int pos = threadIdx.x + i * threads;
        values[i] = input[(pos / spatial4 * 256 + feature) * spatial4 + pos % spatial4];
    }
    const double2 total = reduce_pair(moments(values, origin));
    __shared__ double2 stat;
    if (threadIdx.x == 0) {
        const double mean = total.x / (8 * 4096);
        const double var = fmax(0.0, fma(-mean, mean, total.y / (8 * 4096)));
        stat = make_double2(origin + mean, rsqrt(var + 1e-5));
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < packs; ++i) {
        const int pos = threadIdx.x + i * threads;
        output[(pos / spatial4 * 256 + feature) * spatial4 + pos % spatial4] = apply_stats(values[i], stat);
    }
}

}  // namespace

extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2) {
    const size_t spatial = D1 * D2;
    if (B == 0 || F == 0 || spatial == 0) return;
    if (((reinterpret_cast<uintptr_t>(X) | reinterpret_cast<uintptr_t>(Y)) % 16 == 0) &&
        spatial % packed_elements == 0 && B <= 65535 && F <= 65535) {
        const float4* src = reinterpret_cast<const float4*>(X);
        float4* dst = reinterpret_cast<float4*>(Y);
        if (B == 8 && F == 256 && spatial == 4096) {
            small_channel<<<F, threads>>>(src, dst);
            return;
        }
        const unsigned tiles = spatial / packed_elements;
        const size_t count = B * F * tiles;
        double2* partials;
        cudaMallocAsync(&partials, (count + F) * sizeof(double2), 0);
        double2* stats = partials + count;
        const dim3 grid(tiles, F, B);
        packed_partials<<<grid, threads>>>(src, partials, B, F, spatial / 4);
        channel_stats<<<F, threads>>>(X, partials, stats, B, spatial, tiles);
        packed_output<<<grid, threads>>>(src, dst, stats, F, spatial / 4);
        cudaFreeAsync(partials, 0);
        return;
    }
    const unsigned tiles = (spatial + elements - 1) / elements;
    const size_t count = B * F * tiles;
    double2* partials;
    cudaMallocAsync(&partials, (count + F) * sizeof(double2), 0);
    double2* stats = partials + count;
    partial_moments<<<count, threads>>>(X, partials, B, F, spatial, tiles);
    channel_stats<<<F, threads>>>(X, partials, stats, B, spatial, tiles);
    normalize<<<count, threads>>>(X, Y, stats, F, spatial, tiles);
    cudaFreeAsync(partials, 0);
}
