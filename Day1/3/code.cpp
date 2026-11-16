#include <cuda_runtime.h>

#ifndef ARG_COLUMNS_128
#define ARG_COLUMNS_128 2
#endif
#ifndef ARG_COLUMNS_256
#define ARG_COLUMNS_256 2
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
    return (FindMax ? !(next <= previous) : !(next >= previous)) && !isnan(previous);
}

__device__ __forceinline__ float extreme(float a, float b) {
#if __CUDA_ARCH__ >= 800
    float result;
    if (FindMax) asm("max.NaN.f32 %0,%1,%2;" : "=f"(result) : "f"(a), "f"(b));
    else asm("min.NaN.f32 %0,%1,%2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
#else
    return improves(b, a) ? b : a;
#endif
}

__device__ Candidate choose(Candidate a, Candidate b) {
    const bool tied = a.value == b.value || (isnan(a.value) && isnan(b.value));
    return improves(b.value, a.value) || (tied && b.index < a.index) ? b : a;
}

// 相同的零和 NaN 共用顺序值，最早下标在归约时单独比较。
__device__ unsigned rank_value(float value) {
    unsigned bits = __float_as_uint(FindMax ? value : -value);
    const unsigned magnitude = bits & 0x7fffffffu;
    if (magnitude == 0) bits = 0;
    if (magnitude > 0x7f800000u) return 0xffffffffu;
    return bits & 0x80000000u ? ~bits : (bits ^ 0x80000000u);
}

template<int V>
__device__ __forceinline__ void read_columns(const float* input, float (&values)[V]) {
    if constexpr (V == 4)
        asm("ld.global.cg.v4.f32 {%0,%1,%2,%3},[%4];"
            : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]) : "l"(input));
    else if constexpr (V == 2)
        asm("ld.global.cg.v2.f32 {%0,%1},[%2];" : "=f"(values[0]), "=f"(values[1]) : "l"(input));
    else
        values[0] = *input;
}

template<int Length, int Inner, int V, int Parts>
__global__ void tiled_columns(const float* __restrict__ input, int* __restrict__ output) {
    static_assert(Inner % (32 * V) == 0 && Length % Parts == 0);
    __shared__ float partial_values[V][Parts][32];
    __shared__ int partial_indices[V][Parts][32];
    const int lane = threadIdx.x % 32, part = threadIdx.x / 32;
    const int col = (int(blockIdx.x) * 32 + lane) * V;
    const int base = col / Inner * Length * Inner + col % Inner;
    constexpr int Count = Length / Parts;
    float values[Count][V];
    #pragma unroll
    for (int r = 0; r < Count; ++r)
        read_columns(input + base + (part + r * Parts) * Inner, values[r]);
    float best[V];
    int indices[V];
    #pragma unroll
    for (int j = 0; j < V; ++j) {
        float a = values[0][j], b = values[1][j];
        #pragma unroll
        for (int r = 2; r < Count; r += 2) {
            a = extreme(a, values[r][j]);
            b = extreme(b, values[r + 1][j]);
        }
        best[j] = extreme(a, b);
        int first = Count - 1;
        if (isnan(best[j])) {
            #pragma unroll
            for (int r = Count - 2; r >= 0; --r)
                if (isnan(values[r][j])) first = r;
        } else {
            #pragma unroll
            for (int r = Count - 2; r >= 0; --r)
                if (values[r][j] == best[j]) first = r;
        }
        indices[j] = part + Parts * first;
    }
    #pragma unroll
    for (int j = 0; j < V; ++j) {
        partial_values[j][part][lane] = best[j];
        partial_indices[j][part][lane] = indices[j];
    }
    __syncthreads();
    if (part == 0) {
        #pragma unroll
        for (int j = 0; j < V; ++j) {
            #pragma unroll
            for (int p = 1; p < Parts; ++p) {
                const float v = partial_values[j][p][lane];
                const Candidate selected = choose({best[j], indices[j]}, {v, partial_indices[j][p][lane]});
                best[j] = selected.value;
                indices[j] = selected.index;
            }
            output[col + j] = indices[j];
        }
    }
}

template<int Length, int Width>
__global__ void tiled_rows(const float* __restrict__ input, int* __restrict__ output, int rows) {
    const int row = (int(blockIdx.x) * Threads + int(threadIdx.x)) / Width;
    if (row >= rows) return;
    const int lane = threadIdx.x % Width;
    constexpr int Groups = Length / (4 * Width);
    float values[Groups][4];
    #pragma unroll
    for (int g = 0; g < Groups; ++g)
        read_columns(input + row * Length + (g * Width + lane) * 4, values[g]);
    float local = values[0][0];
    #pragma unroll
    for (int g = 0; g < Groups; ++g) {
        const float a = extreme(values[g][0], values[g][1]);
        const float b = extreme(values[g][2], values[g][3]);
        local = extreme(local, extreme(a, b));
    }
    const unsigned mask = (0xffffffffu >> (32 - Width)) << ((threadIdx.x % 32) & ~(Width - 1));
    unsigned winner = rank_value(local);
#if __CUDA_ARCH__ >= 800
    winner = __reduce_max_sync(mask, winner);
#else
    #pragma unroll
    for (int s = Width / 2; s; s /= 2)
        winner = max(winner, __shfl_xor_sync(mask, winner, s, Width));
#endif
    const unsigned bits = winner & 0x80000000u ? winner ^ 0x80000000u : ~winner;
    const float target = FindMax ? __uint_as_float(bits) : -__uint_as_float(bits);
    int index = Length;
    #pragma unroll
    for (int g = Groups - 1; g >= 0; --g) {
        #pragma unroll
        for (int j = 3; j >= 0; --j)
            if (values[g][j] == target || (winner == 0xffffffffu && isnan(values[g][j])))
                index = (g * Width + lane) * 4 + j;
    }
#if __CUDA_ARCH__ >= 800
    index = __reduce_min_sync(mask, index);
#else
    #pragma unroll
    for (int s = Width / 2; s; s /= 2)
        index = min(index, __shfl_xor_sync(mask, index, s, Width));
#endif
    if (lane == 0) output[row] = index;
}

template<int Length, int Inner, int V, int Columns>
__global__ void warp_columns(const float* __restrict__ input, int* __restrict__ output) {
    constexpr int Groups = 32 / Columns, Count = Length / Groups;
    constexpr int Tile = Columns * V;
    static_assert(Inner % Tile == 0 && Length % Groups == 0);
    const int lane = threadIdx.x % 32, group = lane / Columns;
    const int warp = int(blockIdx.x) * Warps + int(threadIdx.x) / 32;
    const int col = warp * Tile + (lane % Columns) * V;
    const int base = col / Inner * Length * Inner + col % Inner;
    const int first = group * Count;
    float best[V];
    int index[V];
    read_columns(input + base + first * Inner, best);
    #pragma unroll
    for (int j = 0; j < V; ++j) index[j] = first;
    #pragma unroll 8
    for (int r = 1; r < Count; ++r) {
        float values[V];
        read_columns(input + base + (first + r) * Inner, values);
        #pragma unroll
        for (int j = 0; j < V; ++j) {
            if (improves(values[j], best[j])) index[j] = first + r;
            best[j] = extreme(values[j], best[j]);
        }
    }
    #pragma unroll
    for (int step = Columns; step < 32; step *= 2) {
        #pragma unroll
        for (int j = 0; j < V; ++j) {
            const float v = __shfl_down_sync(0xffffffffu, best[j], step);
            const int k = __shfl_down_sync(0xffffffffu, index[j], step);
            if (improves(v, best[j])) index[j] = k;
            best[j] = extreme(best[j], v);
        }
    }
    if (group == 0) {
        #pragma unroll
        for (int j = 0; j < V; ++j) output[col + j] = index[j];
    }
}

template<int Length, int Width>
__global__ void packed_rows(const float* __restrict__ input, int* __restrict__ output, int rows) {
    constexpr int V = Length == 1024 ? 4 : Length / Width;
    constexpr int Block = Length == 1024 ? 256 : Threads;
    const int tid = threadIdx.x;
    const int row = Length == 1024 ? int(blockIdx.x) : (int(blockIdx.x) * Block + tid) / Width;
    if (row >= rows) return;
    const int lane = tid % Width;
    const int start = Length == 1024 ? tid * V : lane * V;
    float values[V];
    read_columns(input + row * Length + start, values);
    float local = values[0];
    #pragma unroll
    for (int j = 1; j < V; ++j) local = extreme(local, values[j]);
    const unsigned mask = (0xffffffffu >> (32 - Width)) << ((tid % 32) & ~(Width - 1));
    const unsigned rank = rank_value(local);
    unsigned winner = rank;
#if __CUDA_ARCH__ >= 800
    winner = __reduce_max_sync(mask, rank);
#else
    #pragma unroll
    for (int s = Width / 2; s; s /= 2)
        winner = max(winner, __shfl_xor_sync(mask, winner, s, Width));
#endif
    const unsigned lanes = __ballot_sync(mask, rank == winner);
    const int first_lane = __ffs(lanes) - 1;
    __shared__ unsigned warp_winners[8];
    __shared__ int warp_indices[8];
    if ((tid & 31) == first_lane) {
        int first = V - 1;
        #pragma unroll
        for (int j = V - 2; j >= 0; --j)
            if (values[j] == local || isnan(values[j])) first = j;
        if constexpr (Length == 64) output[row] = start + first;
        else {
            warp_winners[tid / 32] = winner;
            warp_indices[tid / 32] = start + first;
        }
    }
    if constexpr (Length == 1024) {
        __syncthreads();
        if (tid < 32) {
            unsigned best = warp_winners[tid % 8];
#if __CUDA_ARCH__ >= 800
            const unsigned top = __reduce_max_sync(0xffffffffu, best);
#else
            unsigned top = best;
            #pragma unroll
            for (int s = 4; s; s /= 2) top = max(top, __shfl_xor_sync(0xffffffffu, top, s));
#endif
            const unsigned winners = __ballot_sync(0xffffffffu, best == top);
            if (tid == 0) output[row] = warp_indices[__ffs(winners) - 1];
        }
    }
}

__device__ Candidate warp_reduce(Candidate best) {
#if __CUDA_ARCH__ >= 800
    const unsigned rank = rank_value(best.value);
    const unsigned winner = __reduce_max_sync(0xffffffffu, rank);
    const int first = __reduce_min_sync(0xffffffffu, rank == winner ? best.index : 0x7fffffff);
    const unsigned bits = winner & 0x80000000u ? winner ^ 0x80000000u : ~winner;
    const float value = __uint_as_float(bits);
    return {FindMax ? value : -value, first};
#else
    for (int step = 16; step > 0; step /= 2) {
        const float value = __shfl_down_sync(0xffffffff, best.value, step);
        const int index = __shfl_down_sync(0xffffffff, best.index, step);
        best = choose(best, {value, index});
    }
    return best;
#endif
}

template<int Stride>
__device__ Candidate read_row(const float* row, int length, int start) {
    const float infinity = __int_as_float(0x7f800000);
    Candidate best{FindMax ? -infinity : infinity, length};
    if (length == 1024 && reinterpret_cast<size_t>(row) % alignof(float4) == 0) {
        const float4* vectors = reinterpret_cast<const float4*>(row);
        #pragma unroll
        for (int group = 0; group < 256 / Stride; ++group) {
            const int i = start + group * Stride;
            const float4 v = vectors[i];
            if (i == start || improves(v.x, best.value)) best = {v.x, 4 * i};
            if (improves(v.y, best.value)) best = {v.y, 4 * i + 1};
            if (improves(v.z, best.value)) best = {v.z, 4 * i + 2};
            if (improves(v.w, best.value)) best = {v.w, 4 * i + 3};
        }
        return warp_reduce(best);
    }
    for (int r = start; r < length; r += Stride)
        best = choose(best, {row[r], r});
    return warp_reduce(best);
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

__device__ void reduce_rows(const float* input, int* output, size_t rows, int length) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (length == 64 && reinterpret_cast<size_t>(input) % alignof(float4) == 0) {
        rows64(reinterpret_cast<const float4*>(input), output, rows);
        return;
    }
    if (ARG_LONG_WARPS == 1 || length <= 64) {
        for (size_t row = size_t(blockIdx.x) * Warps + warp; row < rows; row += gridDim.x * Warps) {
            Candidate best = read_row<32>(input + row * length, length, lane);
            if (lane == 0) output[row] = best.index;
        }
        return;
    }
#if ARG_LONG_WARPS == 4
    __shared__ Candidate partial[Warps];
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        Candidate best = read_row<Threads>(input + row * length, length, threadIdx.x);
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

__device__ void columns32x4(const float4* input, int* output, size_t rows) {
    constexpr int Stride = 262144 / 4;
    for (size_t col = size_t(blockIdx.x) * Threads + threadIdx.x; col < rows * Stride;
         col += gridDim.x * Threads) {
        const float4* base = input + col / Stride * 32 * Stride + col % Stride;
        float4 best = base[0];
        int4 indices = make_int4(0, 0, 0, 0);
        #pragma unroll
        for (int r = 1; r < 32; ++r) {
            const float4 v = base[r * Stride];
            if (improves(v.x, best.x)) { best.x = v.x; indices.x = r; }
            if (improves(v.y, best.y)) { best.y = v.y; indices.y = r; }
            if (improves(v.z, best.z)) { best.z = v.z; indices.z = r; }
            if (improves(v.w, best.w)) { best.w = v.w; indices.w = r; }
        }
        output[4 * col] = indices.x;
        output[4 * col + 1] = indices.y;
        output[4 * col + 2] = indices.z;
        output[4 * col + 3] = indices.w;
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

__device__ void reduce_columns(const float* input, int* output, Layout layout) {
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
    else if (length == 32 && inner == 262144 && reinterpret_cast<size_t>(input) % alignof(float4) == 0)
        columns32x4(reinterpret_cast<const float4*>(input), output, rows);
    else if (length == 32 && inner == 262144)
        fixed_columns<32, 262144>(input, output, rows);
    else if (length == 128 && inner == 256)
        short_column_grid(input, output, rows);
    else
        scan_columns(input, output, rows, length, inner);
}

template<int Length, int Inner>
__global__ void fixed_layout(const float* input, int* output, size_t rows) {
    if (Inner == 1)
        reduce_rows(input, output, rows, Length);
    else
        reduce_columns(input, output, {rows, Inner, Length});
}

__global__ void general_layout(const float* input, int* output, const int* shape,
                               int dim, int ndim, Layout layout) {
    if (shape) layout = get_layout(shape, dim, ndim);
    if (layout.inner == 1)
        reduce_rows(input, output, layout.rows, layout.length);
    else
        reduce_columns(input, output, layout);
}

int grid_for(size_t work, int per_block) {
    const size_t blocks = 1 + (work - 1) / per_block;
    return blocks < 2048 ? static_cast<int>(blocks) : 2048;
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
        const size_t rows = layout.rows;
        const size_t outputs = rows * layout.inner;
        if (outputs <= size_t(0x7fffffff / layout.length)) {
            const int n = int(outputs);
            const size_t address = reinterpret_cast<size_t>(input);
            if (layout.length == 128 && layout.inner == 256)
                return warp_columns<128, 256, 1, 8><<<n / 32, Threads>>>(input, output);
            if (layout.length == 32 && layout.inner == 262144 && address % 16 == 0)
                return warp_columns<32, 262144, 4, 8><<<n / 128, Threads>>>(input, output);
            if (ARG_COLUMNS_128 == 2 && layout.length == 128 && layout.inner == 128 && address % 16 == 0)
                return warp_columns<128, 128, 2, 16><<<n / 128, Threads>>>(input, output);
            if (ARG_COLUMNS_256 == 2 && layout.length == 256 && layout.inner == 65536 && address % 16 == 0)
                return warp_columns<256, 65536, 4, 8><<<n / 128, Threads>>>(input, output);
            if (layout.inner == 1 && address % 16 == 0) {
                if (layout.length == 64)
                    return tiled_rows<64, 2><<<(n + 63) / 64, Threads>>>(input, output, n);
                if (ARG_LONG_WARPS == 4 && layout.length == 1024)
                    return tiled_rows<1024, 32><<<(n + 3) / 4, Threads>>>(input, output, n);
            }
        }
        if (layout.length == 64 && layout.inner == 1)
            return fixed_layout<64, 1><<<grid_for(rows, Threads / 16), Threads>>>(input, output, rows);
        if (layout.length == 1024 && layout.inner == 1)
            return fixed_layout<1024, 1><<<grid_for(rows, Warps / ARG_LONG_WARPS), Threads>>>(input, output, rows);
        if (layout.length == 128 && layout.inner == 256)
            return fixed_layout<128, 256><<<grid_for(outputs, Warps * 8), Threads>>>(input, output, rows);
        if (layout.length == 32 && layout.inner == 262144)
            return fixed_layout<32, 262144><<<grid_for(outputs, Threads * 4), Threads>>>(input, output, rows);
        if (layout.length == 128 && layout.inner == 128)
            return fixed_layout<128, 128><<<grid_for(outputs, Threads * ARG_COLUMNS_128), Threads>>>(input, output, rows);
        if (layout.length == 256 && layout.inner == 65536)
            return fixed_layout<256, 65536><<<grid_for(outputs, Threads * ARG_COLUMNS_256), Threads>>>(input, output, rows);
    }
    general_layout<<<512, Threads>>>(input, output, shape, dim, ndim, layout);
}
