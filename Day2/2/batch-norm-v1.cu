#include <cuda_runtime.h>
#include <cstddef>

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

}  // namespace

extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2) {
    const size_t spatial = D1 * D2;
    if (B == 0 || F == 0 || spatial == 0) return;
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
