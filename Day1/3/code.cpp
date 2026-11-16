#include <cuda_runtime.h>

#ifndef ARG_COLUMNS_128
#define ARG_COLUMNS_128 1
#endif
#ifndef ARG_COLUMNS_256
#define ARG_COLUMNS_256 1
#endif
#ifndef ARG_LONG_WARPS
#define ARG_LONG_WARPS 4
#endif

namespace {
constexpr bool FindMax = false;
constexpr int Threads = 128;
constexpr int Warps = Threads / 32;
static_assert(ARG_COLUMNS_128 == 1 || ARG_COLUMNS_128 == 2, "invalid ARG_COLUMNS_128");
static_assert(ARG_COLUMNS_256 == 1 || ARG_COLUMNS_256 == 2, "invalid ARG_COLUMNS_256");
static_assert(ARG_LONG_WARPS == 1 || ARG_LONG_WARPS == Warps, "invalid ARG_LONG_WARPS");

struct Candidate { float value; int index; };

__device__ bool improves(float next, float previous) {
    return (FindMax ? next > previous : next < previous) || (isnan(next) && !isnan(previous));
}

__device__ Candidate choose(Candidate a, Candidate b) {
    const bool tied = a.value == b.value || (isnan(a.value) && isnan(b.value));
    return improves(b.value, a.value) || (tied && b.index < a.index) ? b : a;
}

__device__ Candidate warp_reduce(Candidate best) {
    for (int step = 16; step > 0; step /= 2) {
        const float value = __shfl_down_sync(0xffffffff, best.value, step);
        const int index = __shfl_down_sync(0xffffffff, best.index, step);
        best = choose(best, {value, index});
    }
    return best;
}

__device__ Candidate read_row(const float* row, int length, int start, int stride) {
    const float infinity = __int_as_float(0x7f800000);
    Candidate best{FindMax ? -infinity : infinity, length};
    for (int r = start; r < length; r += stride)
        best = choose(best, {row[r], r});
    return warp_reduce(best);
}

__global__ void contiguous(const float* input, int* output, const int* shape, int dim) {
    size_t rows = 1;
    for (int i = 0; i < dim; ++i) rows *= shape[i];
    const int length = shape[dim], lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (ARG_LONG_WARPS == 1 || length <= 64) {
        for (size_t row = size_t(blockIdx.x) * Warps + warp; row < rows; row += gridDim.x * Warps) {
            Candidate best = read_row(input + row * length, length, lane, 32);
            if (lane == 0) output[row] = best.index;
        }
        return;
    }
#if ARG_LONG_WARPS == 4
    __shared__ Candidate partial[Warps];
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        Candidate best = read_row(input + row * length, length, threadIdx.x, Threads);
        if (lane == 0) partial[warp] = best;
        __syncthreads();
        if (warp == 0) {
            best = warp_reduce(partial[lane % Warps]);
            if (lane == 0) output[row] = best.index;
        }
        __syncthreads();
    }
#endif
}

__device__ void scan_columns(const float* input, int* output, size_t rows, int length, size_t inner) {
    const size_t step = size_t(gridDim.x) * blockDim.x;
    for (size_t out = size_t(blockIdx.x) * blockDim.x + threadIdx.x; out < rows * inner; out += step) {
        const float* base = input + (out / inner) * length * inner + out % inner;
        Candidate best{base[0], 0};
        for (int r = 1; r < length; ++r) {
            const float next = base[size_t(r) * inner];
            if (improves(next, best.value)) best = {next, r};
        }
        output[out] = best.index;
    }
}

__device__ void scan_pairs(const float2* input, int* output, size_t rows, int length, size_t pairs) {
    const size_t step = size_t(gridDim.x) * blockDim.x;
    for (size_t out = size_t(blockIdx.x) * blockDim.x + threadIdx.x; out < rows * pairs; out += step) {
        const float2* base = input + (out / pairs) * length * pairs + out % pairs;
        float2 best = base[0];
        int2 indices = make_int2(0, 0);
        for (int r = 1; r < length; ++r) {
            const float2 next = base[size_t(r) * pairs];
            if (improves(next.x, best.x)) { best.x = next.x; indices.x = r; }
            if (improves(next.y, best.y)) { best.y = next.y; indices.y = r; }
        }
        output[2 * out] = indices.x;
        output[2 * out + 1] = indices.y;
    }
}

__global__ void strided(const float* input, int* output, const int* shape, int dim, int ndim) {
    size_t rows = 1, inner = 1;
    for (int i = 0; i < dim; ++i) rows *= shape[i];
    for (int i = dim + 1; i < ndim; ++i) inner *= shape[i];
    const int length = shape[dim];
    const bool pairs = (ARG_COLUMNS_128 == 2 && length == 128 && inner == 128)
                    || (ARG_COLUMNS_256 == 2 && length == 256 && inner == 65536);
    if (pairs && reinterpret_cast<size_t>(input) % alignof(float2) == 0)
        scan_pairs(reinterpret_cast<const float2*>(input), output, rows, length, inner / 2);
    else
        scan_columns(input, output, rows, length, inner);
}
}

extern "C" void solution(const float* input, int dim, int* output, const int* shape, int ndim) {
    if (dim == ndim - 1)
        contiguous<<<512, Threads>>>(input, output, shape, dim);
    else
        strided<<<512, Threads>>>(input, output, shape, dim, ndim);
}
