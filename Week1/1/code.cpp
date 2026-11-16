#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 32

__global__ void init_kernel(const float* adj_matrix, float* output, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        int idx = i * n + j;
        if (i == j) {
            output[idx] = 0.0f;
        } else if (adj_matrix[idx] == 0.0f) {
            output[idx] = INFINITY;
        } else {
            output[idx] = adj_matrix[idx];
        }
    }
}

__global__ void floyd_kernel(float* output, int n, int k) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int i = by * BLOCK_SIZE + ty;
    int j = bx * BLOCK_SIZE + tx;

    __shared__ float s_dist_ik[BLOCK_SIZE];
    __shared__ float s_dist_kj[BLOCK_SIZE];

    if (tx == 0 && i < n) {
        s_dist_ik[ty] = output[i * n + k];
    }

    if (ty == 0 && j < n) {
        s_dist_kj[tx] = output[k * n + j];
    }
    __syncthreads();

    if (i < n && j < n) {
        float ik = s_dist_ik[ty];
        float kj = s_dist_kj[tx];

        if (ik != INFINITY && kj != INFINITY) {
            float new_dist = ik + kj;
            int idx = i * n + j;
            if (new_dist < output[idx]) {
                output[idx] = new_dist;
            }
        }
    }
}

__global__ void finalize_kernel(float* output, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        int idx = i * n + j;
        if (output[idx] == INFINITY) {
            output[idx] = -1.0f;
        }
    }
}

extern "C" void solution(const float* adj_matrix, float* output, size_t n) {
    if (n == 0) return;

    int n_int = static_cast<int>(n);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n_int + BLOCK_SIZE - 1) / BLOCK_SIZE, (n_int + BLOCK_SIZE - 1) / BLOCK_SIZE);

    init_kernel<<<grid, block>>>(adj_matrix, output, n_int);

    for (int k = 0; k < n_int; ++k) {
        floyd_kernel<<<grid, block>>>(output, n_int, k);
    }

    finalize_kernel<<<grid, block>>>(output, n_int);
}
