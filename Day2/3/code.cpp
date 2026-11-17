#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int tile_width = 96;
constexpr int tile_height = 16;
constexpr int threads = 128;

template<int K>
__global__ void blur_tile(const float* input, float* output, int height, int width) {
    constexpr int radius = K / 2;
    __shared__ float prefixes[tile_height][threads];
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int first_x = blockIdx.x * tile_width;
    const int first_y = blockIdx.y * tile_height;
    const int x = first_x + t - radius;
    const bool valid_x = t < tile_width + K - 1 && unsigned(x) < unsigned(width);

    // 沿列移动窗口。每前进一行，只需加入新像素并减去离开窗口的像素。
    float column = 0;
    if (valid_x) {
        #pragma unroll
        for (int i = -radius; i <= radius; ++i) {
            const int y = first_y + i;
            if (unsigned(y) < unsigned(height)) column += input[size_t(y) * width + x];
        }
    }
    #pragma unroll
    for (int row = 0; row < tile_height; ++row) {
        float prefix = column;
        #pragma unroll
        for (int delta = 1; delta < 32; delta *= 2) {
            const float previous = __shfl_up_sync(0xffffffff, prefix, delta);
            if (lane >= delta) prefix += previous;
        }
        prefixes[row][t] = prefix;
        if (row + 1 < tile_height && valid_x) {
            const int leaving = first_y + row - radius;
            const int entering = first_y + row + radius + 1;
            if (leaving >= 0 && leaving < height) column -= input[size_t(leaving) * width + x];
            if (entering < height) column += input[size_t(entering) * width + x];
        }
    }
    __syncthreads();

    for (int index = t; index < tile_height * tile_width; index += threads) {
        const int row = index / tile_width;
        const int left = index % tile_width;
        const int out_x = first_x + left;
        const int out_y = first_y + row;
        if (out_x >= width || out_y >= height) continue;
        const int right = left + K - 1;
        float sum = prefixes[row][right];
        if (left / 32 != right / 32) sum += prefixes[row][left | 31];
        if (left % 32 != 0) sum -= prefixes[row][left - 1];
        const int nx = min(width - 1, out_x + radius) - max(0, out_x - radius) + 1;
        const int ny = min(height - 1, out_y + radius) - max(0, out_y - radius) + 1;
        output[size_t(out_y) * width + out_x] = sum / float(nx * ny);
    }
}

template<int K>
__global__ void blur_rows(const float* input, float* output, int height, int width) {
    constexpr int radius = K / 2;
    constexpr int rows = 32;
    __shared__ float4 horizontal[rows + K - 1][tile_width / 4];
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int first_x = blockIdx.x * tile_width;
    const int first_y = blockIdx.y * rows;
    const int input_x = first_x - 16 + 4 * lane;

    // 一个 warp 连续读取 128 列，居中的 96 列各自产生一个水平方向的窗口和。
    for (int row = warp; row < rows + K - 1; row += 6) {
        const int y = first_y + row - radius;
        float4 v = make_float4(0, 0, 0, 0);
        if (unsigned(y) < unsigned(height) && unsigned(input_x) < unsigned(width)) {
            v = reinterpret_cast<const float4*>(input + size_t(y) * width)[input_x / 4];
        }
        float sum = (v.x + v.y) + (v.z + v.w);
        #pragma unroll
        for (int delta = 1; delta < 32; delta *= 2) {
            const float prev = __shfl_up_sync(0xffffffff, sum, delta);
            if (lane >= delta) sum += prev;
        }
        float start = __shfl_up_sync(0xffffffff, sum, 1);
        if (lane == 0) start = 0;
        const float prefix[4] = {start + v.x, start + (v.x + v.y),
                                 start + ((v.x + v.y) + v.z), sum};
        float window[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int lo = 16 + i - radius - 1;
            const int hi = 16 + i + radius;
            const float a = __shfl_sync(0xffffffff, prefix[lo % 4], (lane + lo / 4) % 32);
            const float b = __shfl_sync(0xffffffff, prefix[hi % 4], (lane + hi / 4) % 32);
            window[i] = b - a;
        }
        if (lane < tile_width / 4) {
            horizontal[row][lane] = make_float4(window[0], window[1], window[2], window[3]);
        }
    }
    __syncthreads();

    const int thread = warp * 32 + lane;
    const int column = thread % tile_width;
    const int x = first_x + column;
    const int group_y = thread / tile_width * (rows / 2);
    if (x >= width) return;
    const int nx = min(width - 1, x + radius) - max(0, x - radius) + 1;
    const float* strips = reinterpret_cast<const float*>(horizontal) + column;
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < K; ++i) sum += strips[(group_y + i) * tile_width];
    #pragma unroll
    for (int i = 0; i < rows / 2; ++i) {
        const int y = first_y + group_y + i;
        if (y < height) {
            const int ny = min(height - 1, y + radius) - max(0, y - radius) + 1;
            output[size_t(y) * width + x] = sum / float(nx * ny);
        }
        if (i + 1 < rows / 2) {
            sum += strips[(group_y + i + K) * tile_width] - strips[(group_y + i) * tile_width];
        }
    }
}

template<bool interior>
__device__ __forceinline__ float4 read_pixel_group(const float* input, int y, int x,
                                                   int height, int width) {
    if (!interior && (unsigned(x) >= unsigned(width) || unsigned(y) >= unsigned(height))) {
        return make_float4(0, 0, 0, 0);
    }
    return reinterpret_cast<const float4*>(input + size_t(y) * width)[x / 4];
}

template<int K, int rows, bool interior>
__device__ __forceinline__ void warp_strip(const float* input, float* output, int height, int width,
                                           int first_y, int first_x) {
    constexpr int r = K / 2;
    static_assert(rows - 1 <= K);
    const int lane = threadIdx.x;
    const int input_x = first_x - 16 + 4 * lane;
    const int out_x = first_x + 4 * lane;
    float4 column = make_float4(0, 0, 0, 0);
    float4 leaving[rows - 1];
    #pragma unroll
    for (int i = 0; i < K; ++i) {
        const float4 v = read_pixel_group<interior>(input, first_y - r + i, input_x, height, width);
        if (i < rows - 1) leaving[i] = v;
        column.x += v.x; column.y += v.y; column.z += v.z; column.w += v.w;
    }
    #pragma unroll
    for (int row = 0; row < rows; ++row) {
        const float first_two = column.x + column.y;
        const float last_two = column.z + column.w;
        const float group = first_two + last_two;
        const float front[5] = {0, column.x, first_two, first_two + column.z, group};
        const float back[5] = {0, column.w, last_two, column.y + last_two, group};
        constexpr int first_full = (16 + 3 - r + 3) / 4;
        constexpr int past_full = (16 + r + 1) / 4;
        float common = 0;
        #pragma unroll
        for (int g = first_full; g < past_full; ++g) {
            common += __shfl_sync(0xffffffff, group, (lane + g) % 32);
        }
        float window[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int left_count = 4 * first_full - (16 + i - r);
            const int right_count = 16 + i + r + 1 - 4 * past_full;
            float sum = common;
            if (left_count > 0) sum += __shfl_sync(0xffffffff, back[min(left_count, 4)], (lane + first_full - 1) % 32);
            if (left_count > 4) sum += __shfl_sync(0xffffffff, back[left_count - 4], (lane + first_full - 2) % 32);
            if (right_count > 0) sum += __shfl_sync(0xffffffff, front[min(right_count, 4)], (lane + past_full) % 32);
            if (right_count > 4) sum += __shfl_sync(0xffffffff, front[right_count - 4], (lane + past_full + 1) % 32);
            if (interior) {
                window[i] = sum * (1.0f / (K * K));
            } else {
                const int y = first_y + row, x = out_x + i;
                const int nx = min(width - 1, x + r) - max(0, x - r) + 1;
                const int ny = min(height - 1, y + r) - max(0, y - r) + 1;
                window[i] = __fdividef(sum, float(max(1, nx * ny)));
            }
        }
        if (lane < 24 && (interior || (out_x < width && first_y + row < height))) {
            float4* dst = reinterpret_cast<float4*>(output + size_t(first_y + row) * width);
            dst[out_x / 4] = make_float4(window[0], window[1], window[2], window[3]);
        }
        if (row + 1 < rows) {
            const float4 v = read_pixel_group<interior>(input, first_y + row + r + 1, input_x, height, width);
            const float4 old = leaving[row];
            column.x += v.x - old.x; column.y += v.y - old.y;
            column.z += v.z - old.z; column.w += v.w - old.w;
        }
    }
}

template<int K, int rows>
__global__ void blur_strips(const float* input, float* output, int height, int width) {
    const int first_x = blockIdx.x * tile_width;
    const int first_y = (blockIdx.y * blockDim.y + threadIdx.y) * rows;
    if (first_y >= height) return;
    if (first_x >= 16 && first_x + 112 <= width &&
        first_y >= K / 2 && first_y + rows + K / 2 <= height) {
        warp_strip<K, rows, true>(input, output, height, width, first_y, first_x);
    } else {
        warp_strip<K, rows, false>(input, output, height, width, first_y, first_x);
    }
}

__global__ void blur_general(const float* input, float* output, size_t radius,
                             size_t height, size_t width) {
    const size_t count = height * width;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += size_t(gridDim.x) * blockDim.x) {
        const size_t y = index / width;
        const size_t x = index % width;
        const size_t top = y > radius ? y - radius : 0;
        const size_t left = x > radius ? x - radius : 0;
        const size_t bottom = y + radius + 1 < height ? y + radius + 1 : height;
        const size_t right = x + radius + 1 < width ? x + radius + 1 : width;
        double sum = 0;
        for (size_t i = top; i < bottom; ++i) {
            for (size_t j = left; j < right; ++j) sum += input[i * width + j];
        }
        output[index] = float(sum / double((bottom - top) * (right - left)));
    }
}

}  // namespace

extern "C" void solution(const float* input_image, int kernel_size, float* output_image,
                         size_t height, size_t width) {
    if (height == 0 || width == 0 || kernel_size <= 0) return;
    const size_t rows = (height + tile_height - 1) / tile_height;
    if (rows <= 65535 && width < INT_MAX - threads) {
        if (width % 4 == 0 &&
            ((reinterpret_cast<uintptr_t>(input_image) | reinterpret_cast<uintptr_t>(output_image)) % 16 == 0)) {
            constexpr int strip_rows = 12, warps = 4;
            const dim3 grid((width + tile_width - 1) / tile_width, (height + strip_rows * warps - 1) / (strip_rows * warps));
            const dim3 block(32, warps);
            if (kernel_size == 15) {
                blur_strips<15, strip_rows><<<grid, block>>>(input_image, output_image, height, width);
                return;
            }
            if (kernel_size == 21) {
                blur_strips<21, strip_rows><<<grid, block>>>(input_image, output_image, height, width);
                return;
            }
            if (kernel_size == 27) {
                blur_strips<27, strip_rows><<<grid, block>>>(input_image, output_image, height, width);
                return;
            }
        }
        if (width % 4 == 0 && reinterpret_cast<uintptr_t>(input_image) % 16 == 0) {
            const dim3 grid((width + tile_width - 1) / tile_width, (height + 31) / 32);
            const dim3 block(32, 6);
            if (kernel_size == 15) {
                blur_rows<15><<<grid, block>>>(input_image, output_image, height, width);
                return;
            }
            if (kernel_size == 21) {
                blur_rows<21><<<grid, block>>>(input_image, output_image, height, width);
                return;
            }
            if (kernel_size == 27) {
                blur_rows<27><<<grid, block>>>(input_image, output_image, height, width);
                return;
            }
        }
        const dim3 grid((width + tile_width - 1) / tile_width, rows);
        if (kernel_size == 15) {
            blur_tile<15><<<grid, threads>>>(input_image, output_image, height, width);
            return;
        }
        if (kernel_size == 21) {
            blur_tile<21><<<grid, threads>>>(input_image, output_image, height, width);
            return;
        }
        if (kernel_size == 27) {
            blur_tile<27><<<grid, threads>>>(input_image, output_image, height, width);
            return;
        }
    }
    const size_t blocks = (height * width + 255) / 256;
    blur_general<<<blocks < 65535 ? blocks : 65535, 256>>>(
        input_image, output_image, kernel_size / 2, height, width);
}
