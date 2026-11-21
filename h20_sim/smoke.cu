// Independent CUDA/SM90 launch test with an odd-sized integer vector.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { const auto status = (call); if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s failed: %d\n", #call, int(status)); return 2; } } while (0)

#ifndef H20_SIM_HOST
extern "C" __global__ void vector_add(const int* a, const int* b, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
#else
#define DRIVER(call) do { const auto status = (call); if (status != CUDA_SUCCESS) { \
    std::fprintf(stderr, "%s failed: %d\n", #call, int(status)); return 2; } } while (0)
int main() {
    constexpr int count = 37;
    int n = count;
    int a[count], b[count], result[count];
    for (int i = 0; i < n; ++i) { a[i] = i * 7 - 100; b[i] = 3 - i * 2; }
    int *da, *db, *out;
    CHECK(cudaMalloc(&da, sizeof(a)));
    CHECK(cudaMalloc(&db, sizeof(b)));
    CHECK(cudaMalloc(&out, sizeof(result)));
    CHECK(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));
    CUmodule module;
    CUfunction kernel;
    DRIVER(cuModuleLoad(&module, "smoke.cubin"));
    DRIVER(cuModuleGetFunction(&kernel, module, "vector_add"));
    void* parameters[] = {&da, &db, &out, &n};
    DRIVER(cuLaunchKernel(kernel, 1, 1, 1, 64, 1, 1, 0, nullptr, parameters, nullptr));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(result, out, sizeof(result), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) if (result[i] != a[i] + b[i]) return 3;
    CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(out));
    std::puts("SM90_VECTOR_ADD_PASS 37/37");
    return 0;
}
#endif
