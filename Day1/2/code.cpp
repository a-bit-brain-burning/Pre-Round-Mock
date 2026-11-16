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
constexpr bool FindMax = true;
constexpr int Threads = 128;
constexpr int Warps = Threads / 32;
static_assert(ARG_COLUMNS_128 == 1 || ARG_COLUMNS_128 == 2, "invalid ARG_COLUMNS_128");
static_assert(ARG_COLUMNS_256 == 1 || ARG_COLUMNS_256 == 2, "invalid ARG_COLUMNS_256");
static_assert(ARG_LONG_WARPS == 1 || ARG_LONG_WARPS == Warps, "invalid ARG_LONG_WARPS");

struct Candidate { float value; int index; };

struct Layout {
    size_t rows;
    size_t inner;
    int length;
};

__host__ __device__ Layout get_layout(const int* shape, int dim, int ndim) {
    Layout layout{1, 1, shape[dim]};
    for (int i = 0; i < dim; ++i) layout.rows *= shape[i];
    for (int i = dim + 1; i < ndim; ++i) layout.inner *= shape[i];
    return layout;
}

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
    if (length == 1024 && reinterpret_cast<size_t>(row) % alignof(float4) == 0) {
        const float4* vectors = reinterpret_cast<const float4*>(row);
        for (int i = start; i < 256; i += stride) {
            const float4 v = vectors[i];
            if (i == start || improves(v.x, best.value)) best = {v.x, 4 * i};
            if (improves(v.y, best.value)) best = {v.y, 4 * i + 1};
            if (improves(v.z, best.value)) best = {v.z, 4 * i + 2};
            if (improves(v.w, best.value)) best = {v.w, 4 * i + 3};
        }
        return warp_reduce(best);
    }
    for (int r = start; r < length; r += stride)
        best = choose(best, {row[r], r});
    return warp_reduce(best);
}

// 将局部极值转换为可由 warp 整数指令比较的顺序，同值保留首次下标。
__device__ unsigned rank_value(float value) {
    unsigned bits = __float_as_uint(FindMax ? value : -value);
    const unsigned magnitude = bits & 0x7fffffffu;
    if (magnitude == 0) bits = 0;
    if (magnitude > 0x7f800000u) return 0xffffffffu;
    return bits & 0x80000000u ? ~bits : (bits ^ 0x80000000u);
}

__device__ void rows64(const float4* input, int* output, size_t rows) {
    const unsigned lane = threadIdx.x % 16;
    const unsigned mask = threadIdx.x % 32 < 16 ? 0xffffu : 0xffff0000u;
    for (size_t row = size_t(blockIdx.x) * (Threads / 16) + threadIdx.x / 16;
         row < rows; row += gridDim.x * (Threads / 16)) {
        const float4 v = input[row * 16 + lane];
        Candidate best{v.x, int(lane * 4)};
        if (improves(v.y, best.value)) best = {v.y, int(lane * 4 + 1)};
        if (improves(v.z, best.value)) best = {v.z, int(lane * 4 + 2)};
        if (improves(v.w, best.value)) best = {v.w, int(lane * 4 + 3)};
        const unsigned rank = rank_value(best.value);
#if __CUDA_ARCH__ >= 800
        const unsigned winner = __reduce_max_sync(mask, rank);
        const int index = __reduce_min_sync(mask, rank == winner ? best.index : 64);
#else
        unsigned winner = rank;
        for (int step = 8; step > 0; step /= 2)
            winner = max(winner, __shfl_xor_sync(mask, winner, step, 16));
        int index = rank == winner ? best.index : 64;
        for (int step = 8; step > 0; step /= 2)
            index = min(index, __shfl_xor_sync(mask, index, step, 16));
#endif
        if (lane == 0) output[row] = index;
    }
}

__global__ void contiguous(const float* input, int* output, const int* shape, int dim, Layout layout) {
    if (shape) layout = get_layout(shape, dim, dim + 1);
    const size_t rows = layout.rows;
    const int length = layout.length, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (length == 64 && reinterpret_cast<size_t>(input) % alignof(float4) == 0) {
        rows64(reinterpret_cast<const float4*>(input), output, rows);
        return;
    }
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

__device__ void short_column_grid(const float* input, int* output, size_t rows) {
    constexpr int Inner = 256, Length = 128;
    const int group = threadIdx.x % 32 / 8;
    const int lane = threadIdx.x % 8;
    for (size_t tile = size_t(blockIdx.x) * Warps + threadIdx.x / 32;
         tile < rows * (Inner / 8); tile += gridDim.x * Warps) {
        const size_t col = tile * 8 + lane;
        const float* base = input + col / Inner * Length * Inner + col % Inner;
        Candidate best{base[group * Inner], group};
        #pragma unroll 4
        for (int i = group + 4; i < Length; i += 4) {
            const float v = base[i * Inner];
            if (improves(v, best.value)) best = {v, i};
        }
        for (int step = 16; step >= 8; step /= 2) {
            Candidate other{__shfl_down_sync(0xffffffffu, best.value, step),
                            __shfl_down_sync(0xffffffffu, best.index, step)};
            best = choose(best, other);
        }
        if (group == 0) output[col] = best.index;
    }
}

template<int Length, int Inner>
__device__ void fixed_columns(const float* input, int* output, size_t rows) {
    for (size_t col = size_t(blockIdx.x) * Threads + threadIdx.x; col < rows * Inner;
         col += gridDim.x * Threads) {
        const float* base = input + (col / Inner) * Length * Inner + col % Inner;
        Candidate best{base[0], 0};
        #pragma unroll 8
        for (int r = 1; r < Length; ++r) {
            const float value = base[r * Inner];
            if (improves(value, best.value)) best = {value, r};
        }
        output[col] = best.index;
    }
}

__global__ void strided(const float* input, int* output, const int* shape, int dim, int ndim, Layout layout) {
    if (shape) layout = get_layout(shape, dim, ndim);
    const size_t rows = layout.rows, inner = layout.inner;
    const int length = layout.length;
    const bool pairs = (ARG_COLUMNS_128 == 2 && length == 128 && inner == 128)
                    || (ARG_COLUMNS_256 == 2 && length == 256 && inner == 65536);
    if (pairs && reinterpret_cast<size_t>(input) % alignof(float2) == 0)
        scan_pairs(reinterpret_cast<const float2*>(input), output, rows, length, inner / 2);
    else if (length == 128 && inner == 128)
        fixed_columns<128, 128>(input, output, rows);
    else if (length == 256 && inner == 65536)
        fixed_columns<256, 65536>(input, output, rows);
    else if (length == 32 && inner == 262144)
        fixed_columns<32, 262144>(input, output, rows);
    else if (length == 128 && inner == 256)
        short_column_grid(input, output, rows);
    else
        scan_columns(input, output, rows, length, inner);
}
}

extern "C" void solution(const float* input, int dim, int* output, const int* shape, int ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return;
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, shape) != cudaSuccess) return;
    Layout layout{};
    if (attributes.type == cudaMemoryTypeUnregistered || attributes.type == cudaMemoryTypeHost) {
        layout = get_layout(shape, dim, ndim);
        if (layout.rows == 0 || layout.inner == 0 || layout.length == 0) return;
        shape = nullptr;
    }
    if (dim == ndim - 1)
        contiguous<<<512, Threads>>>(input, output, shape, dim, layout);
    else
        strided<<<512, Threads>>>(input, output, shape, dim, ndim, layout);
}
