#include <cuda_runtime.h>

#ifndef FW_TILE
#define FW_TILE 64
#endif
#ifndef FW_DPX
#define FW_DPX 0
#endif

namespace {
constexpr int B = FW_TILE;
constexpr int Side = 16;
constexpr int Count = B / Side;
static_assert(B == 32 || B == 64, "FW_TILE must be 32 or 64");
static_assert(FW_DPX == 0 || FW_DPX == 1, "FW_DPX must be 0 or 1");

__device__ float positive_infinity() { return __int_as_float(0x7f800000); }

__global__ void initialize(const float* input, float* dist, size_t n) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n * n) return;
    const float weight = input[index];
    dist[index] = index / n == index % n ? 0.0f : (weight == 0.0f ? positive_infinity() : weight);
}

__global__ void finish(float* dist, size_t elements) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements && dist[index] == positive_infinity()) dist[index] = -1.0f;
}

__device__ float read_cell(const float* dist, size_t n, size_t row, size_t col) {
    return row < n && col < n ? dist[row * n + col] : positive_infinity();
}

__global__ void diagonal(float* dist, size_t n, int pivot) {
    __shared__ float tile[B][B];
    const size_t origin = size_t(pivot) * B;
    for (int y = threadIdx.y; y < B; y += Side)
        for (int x = threadIdx.x; x < B; x += Side)
            tile[y][x] = read_cell(dist, n, origin + y, origin + x);
    __syncthreads();
    for (int k = 0; k < B; ++k) {
        for (int y = threadIdx.y; y < B; y += Side)
            for (int x = threadIdx.x; x < B; x += Side)
                if (y != k && x != k)
                    tile[y][x] = fminf(tile[y][x], tile[y][k] + tile[k][x]);
        __syncthreads();
    }
    for (int y = threadIdx.y; y < B; y += Side)
        for (int x = threadIdx.x; x < B; x += Side)
            if (origin + y < n && origin + x < n)
                dist[(origin + y) * n + origin + x] = tile[y][x];
}

union Cell { float weight; int integer; };

struct FloatOps {
    using Value = float;
    __device__ static Value infinity() { return positive_infinity(); }
    __device__ static Value read(Cell cell) { return cell.weight; }
    __device__ static Value relax(Value a, Value b, Value old) { return fminf(a + b, old); }
    __device__ static float decode(Value value) { return value; }
};

#if FW_DPX
struct DpxOps {
    using Value = int;
    __device__ static Value infinity() { return 1 << 29; }
    __device__ static int encode(float value) { return value == positive_infinity() ? infinity() : int(value); }
    __device__ static Value read(Cell cell) { return cell.integer; }
    __device__ static Value relax(Value a, Value b, Value old) { return __viaddmin_s32(a, b, old); }
    __device__ static float decode(Value value) { return value == infinity() ? positive_infinity() : float(value); }
};
#endif

template<class Ops>
__device__ void update_output(const Cell (&left)[B][B], const Cell (&right)[B][B],
                              float* dist, size_t n, size_t row, size_t col) {
    using Value = typename Ops::Value;
    Value best[Count][Count];
    #pragma unroll
    for (int i = 0; i < Count; ++i)
        #pragma unroll
        for (int j = 0; j < Count; ++j)
            best[i][j] = Ops::infinity();
    for (int k = 0; k < B; ++k) {
        Value a[Count], b[Count];
        #pragma unroll
        for (int i = 0; i < Count; ++i) {
            a[i] = Ops::read(left[threadIdx.y + i * Side][k]);
            b[i] = Ops::read(right[k][threadIdx.x + i * Side]);
        }
        #pragma unroll
        for (int i = 0; i < Count; ++i)
            #pragma unroll
            for (int j = 0; j < Count; ++j)
                best[i][j] = Ops::relax(a[i], b[j], best[i][j]);
    }
    for (int i = 0; i < Count; ++i) {
        const size_t y = row + threadIdx.y + i * Side;
        for (int j = 0; j < Count; ++j) {
            const size_t x = col + threadIdx.x + j * Side;
            if (y < n && x < n)
                dist[y * n + x] = fminf(dist[y * n + x], Ops::decode(best[i][j]));
        }
    }
}

template<bool Panel>
__global__ void update_tiles(float* dist, size_t n, int pivot) {
    const int bx = int(blockIdx.x) + (int(blockIdx.x) >= pivot);
    const int by = int(blockIdx.y) + (int(blockIdx.y) >= pivot);
    const size_t row = size_t(Panel ? (blockIdx.y == 0 ? pivot : bx) : by) * B;
    const size_t col = size_t(Panel && blockIdx.y == 1 ? pivot : bx) * B;
    const size_t origin = size_t(pivot) * B;
    __shared__ Cell left[B][B], right[B][B];
    for (int y = threadIdx.y; y < B; y += Side)
        for (int x = threadIdx.x; x < B; x += Side) {
            left[y][x].weight = read_cell(dist, n, row + y, origin + x);
            right[y][x].weight = read_cell(dist, n, origin + y, col + x);
        }
#if FW_DPX
    if (!Panel) {
        // 题面保证有限边权为整数。两个不超过 2^23 的操作数之和可由 FP32 精确表示。
        bool wide = false;
        for (int y = threadIdx.y; y < B; y += Side)
            for (int x = threadIdx.x; x < B; x += Side) {
                const float a = left[y][x].weight, b = right[y][x].weight;
                wide |= (isfinite(a) && a > 8388608.0f) || (isfinite(b) && b > 8388608.0f);
            }
        if (!__syncthreads_or(wide)) {
            for (int y = threadIdx.y; y < B; y += Side)
                for (int x = threadIdx.x; x < B; x += Side) {
                    left[y][x].integer = DpxOps::encode(left[y][x].weight);
                    right[y][x].integer = DpxOps::encode(right[y][x].weight);
                }
            __syncthreads();
            update_output<DpxOps>(left, right, dist, n, row, col);
            return;
        }
    }
#endif
    __syncthreads();
    update_output<FloatOps>(left, right, dist, n, row, col);
}
}

extern "C" void solution(const float* adj_matrix, float* output, size_t n) {
    if (n == 0) return;
    const unsigned blocks = unsigned((n * n + 255) / 256);
    const int tiles = int((n + B - 1) / B);
    const dim3 threads(Side, Side);
    initialize<<<blocks, 256>>>(adj_matrix, output, n);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        diagonal<<<1, threads>>>(output, n, pivot);
        if (tiles > 1) {
            update_tiles<true><<<dim3(tiles - 1, 2), threads>>>(output, n, pivot);
            update_tiles<false><<<dim3(tiles - 1, tiles - 1), threads>>>(output, n, pivot);
        }
    }
    finish<<<blocks, 256>>>(output, n * n);
}
