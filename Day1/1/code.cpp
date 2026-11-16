#include <cuda_runtime.h>
#include <math.h>

namespace {
constexpr int TILE = 64;
constexpr int ROWS = 8;

__global__ void initialize(const float* input, float* dist, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n * n) {
        const float value = input[index];
        dist[index] = index / n == index % n ? 0.0f
                    : value == 0.0f ? INFINITY : value;
    }
}

// Close the pivot tile before any other tile uses it.
__global__ void close_pivot(float* dist, int n, int pivot) {
    __shared__ float tile[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int start = pivot * TILE;
    #pragma unroll
    for (int r = 0; r < TILE / ROWS; ++r) {
        const int i = start + y + r * ROWS;
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int j = start + x + c * 32;
            tile[y + r * ROWS][x + c * 32] =
                i < n && j < n ? dist[i * n + j] : INFINITY;
        }
    }
    __syncthreads();

    #pragma unroll 1
    for (int k = 0; k < TILE; ++k) {
        const float right0 = tile[k][x];
        const float right1 = tile[k][x + 32];
        #pragma unroll
        for (int r = 0; r < TILE / ROWS; ++r) {
            const int row = y + r * ROWS;
            const float left = tile[row][k];
            // The pivot row and column cannot improve with positive weights.
            // Skipping their writes also avoids shared-memory data races.
            if (row != k) {
                if (x != k)
                    tile[row][x] = fminf(tile[row][x], left + right0);
                if (x + 32 != k)
                    tile[row][x + 32] = fminf(tile[row][x + 32], left + right1);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < TILE / ROWS; ++r) {
        const int i = start + y + r * ROWS;
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int j = start + x + c * 32;
            if (i < n && j < n)
                dist[i * n + j] = tile[y + r * ROWS][x + c * 32];
        }
    }
}

// Update the other tiles in the pivot row and pivot column.
__global__ void close_cross(float* dist, int n, int pivot) {
    __shared__ float diagonal[TILE][TILE];
    __shared__ float tile[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int other = blockIdx.x + (blockIdx.x >= pivot);
    const bool in_row = blockIdx.y == 0;
    const int row_start = (in_row ? pivot : other) * TILE;
    const int col_start = (in_row ? other : pivot) * TILE;
    const int pivot_start = pivot * TILE;

    #pragma unroll
    for (int r = 0; r < TILE / ROWS; ++r) {
        const int row = y + r * ROWS;
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int col = x + c * 32;
            diagonal[row][col] = pivot_start + row < n && pivot_start + col < n
                ? dist[(pivot_start + row) * n + pivot_start + col] : INFINITY;
            tile[row][col] = row_start + row < n && col_start + col < n
                ? dist[(row_start + row) * n + col_start + col] : INFINITY;
        }
    }
    __syncthreads();

    #pragma unroll 1
    for (int k = 0; k < TILE; ++k) {
        const float right0 = in_row ? tile[k][x] : diagonal[k][x];
        const float right1 = in_row ? tile[k][x + 32] : diagonal[k][x + 32];
        #pragma unroll
        for (int r = 0; r < TILE / ROWS; ++r) {
            const int row = y + r * ROWS;
            const float left = in_row ? diagonal[row][k] : tile[row][k];
            if (!in_row || row != k) {
                if (in_row || x != k)
                    tile[row][x] = fminf(tile[row][x], left + right0);
                if (in_row || x + 32 != k)
                    tile[row][x + 32] = fminf(tile[row][x + 32], left + right1);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < TILE / ROWS; ++r) {
        const int row = y + r * ROWS;
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int col = x + c * 32;
            if (row_start + row < n && col_start + col < n)
                dist[(row_start + row) * n + col_start + col] = tile[row][col];
        }
    }
}

// Each thread keeps an 8 x 2 patch of output distances in registers.
__global__ void close_remaining(float* dist, int n, int pivot) {
    __shared__ float left_tile[TILE][TILE];
    __shared__ float right_tile[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int row_start = (blockIdx.y + (blockIdx.y >= pivot)) * TILE;
    const int col_start = (blockIdx.x + (blockIdx.x >= pivot)) * TILE;
    const int pivot_start = pivot * TILE;
    float best[TILE / ROWS][2];

    #pragma unroll
    for (int r = 0; r < TILE / ROWS; ++r) {
        const int row = y + r * ROWS;
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int col = x + c * 32;
            left_tile[row][col] = row_start + row < n && pivot_start + col < n
                ? dist[(row_start + row) * n + pivot_start + col] : INFINITY;
            right_tile[row][col] = pivot_start + row < n && col_start + col < n
                ? dist[(pivot_start + row) * n + col_start + col] : INFINITY;
            best[r][c] = row_start + row < n && col_start + col < n
                ? dist[(row_start + row) * n + col_start + col] : INFINITY;
        }
    }
    __syncthreads();

    #pragma unroll 1
    for (int k = 0; k < TILE; ++k) {
        const float right0 = right_tile[k][x];
        const float right1 = right_tile[k][x + 32];
        #pragma unroll
        for (int r = 0; r < TILE / ROWS; ++r) {
            const float left = left_tile[y + r * ROWS][k];
            best[r][0] = fminf(best[r][0], left + right0);
            best[r][1] = fminf(best[r][1], left + right1);
        }
    }

    #pragma unroll
    for (int r = 0; r < TILE / ROWS; ++r) {
        const int i = row_start + y + r * ROWS;
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int j = col_start + x + c * 32;
            if (i < n && j < n) dist[i * n + j] = best[r][c];
        }
    }
}

__global__ void finalize(float* dist, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count && dist[index] == INFINITY) dist[index] = -1.0f;
}
}  // namespace

extern "C" void solution(const float* adj_matrix, float* output, size_t n) {
    if (n == 0) return;
    const int size = static_cast<int>(n);
    const int tiles = (size + TILE - 1) / TILE;
    const dim3 block(32, ROWS);
    initialize<<<(size * size + 255) / 256, 256>>>(adj_matrix, output, size);
    for (int pivot = 0; pivot < tiles; ++pivot) {
        close_pivot<<<1, block>>>(output, size, pivot);
        if (tiles > 1) {
            close_cross<<<dim3(tiles - 1, 2), block>>>(output, size, pivot);
            close_remaining<<<dim3(tiles - 1, tiles - 1), block>>>(output, size, pivot);
        }
    }
    finalize<<<(size * size + 255) / 256, 256>>>(output, size * size);
}
