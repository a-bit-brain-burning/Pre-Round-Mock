// SM90 SwiGLU replay: execute compiled PTX through the simulator CUDA API.
// Inputs and outputs are raw bytes; numerical checking remains in Python.
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void driver(CUresult result, const char* expression) {
    if (result != CUDA_SUCCESS)
        throw std::runtime_error(std::string(expression) + " failed: " + std::to_string(result));
}
static void runtime(cudaError_t result, const char* expression) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(expression) + " failed: " + std::to_string(result));
}
#define DRV(expr) driver((expr), #expr)
#define RT(expr) runtime((expr), #expr)

int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::runtime_error("usage: replay module entry n element_bytes tile shared expected_library");
        const size_t n = std::stoull(argv[3]), element_bytes = std::stoull(argv[4]);
        const unsigned tile = std::stoul(argv[5]), shared = std::stoul(argv[6]);
        if (!n || !tile || (element_bytes != 2 && element_bytes != 4))
            throw std::runtime_error("invalid replay dimensions");
        Dl_info info{};
        if (!dladdr(reinterpret_cast<void*>(cuLaunchKernel), &info) || !info.dli_fname ||
            std::string(info.dli_fname) != argv[7])
            throw std::runtime_error("CUDA symbol did not resolve to the expected simulator library");
        std::cout << "SIMULATOR_LIBRARY=" << info.dli_fname << '\n';
        // Use the implemented single-device runtime. Explicit context lifecycle
        // APIs in this pinned upstream are placeholders, so do not call them.
        RT(cudaSetDevice(0));
        CUmodule module;
        CUfunction kernel;
        DRV(cuModuleLoad(&module, argv[1]));
        DRV(cuModuleGetFunction(&kernel, module, argv[2]));
        if (shared) DRV(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared));

        const char* names[] = {"a", "b", "g", "da", "db"};
        const size_t bytes = (n + 32) * element_bytes;
        void* pointers[5]{};
        for (unsigned i = 0; i < 5; ++i) {
            std::ifstream input(std::string(names[i]) + ".bin", std::ios::binary | std::ios::ate);
            if (!input || input.tellg() != static_cast<std::streamoff>(bytes))
                throw std::runtime_error("input byte count mismatch");
            input.seekg(0);
            std::vector<char> data(bytes);
            if (!input.read(data.data(), bytes)) throw std::runtime_error("input read failed");
            RT(cudaMalloc(&pointers[i], bytes));
            RT(cudaMemcpy(pointers[i], data.data(), bytes, cudaMemcpyHostToDevice));
        }
        void* global_scratch = nullptr;
        void* profile_scratch = nullptr;
        void* args[] = {&pointers[0], &pointers[1], &pointers[2], &pointers[3], &pointers[4],
                       &global_scratch, &profile_scratch};
        DRV(cuLaunchKernel(kernel, (n + tile - 1) / tile, 1, 1, 128, 1, 1, shared, nullptr, args, nullptr));
        RT(cudaDeviceSynchronize());
        for (unsigned i = 0; i < 5; ++i) {
            std::vector<char> data(bytes);
            RT(cudaMemcpy(data.data(), pointers[i], bytes, cudaMemcpyDeviceToHost));
            std::ofstream output(std::string(names[i]) + ".out.bin", std::ios::binary);
            if (!output.write(data.data(), bytes)) throw std::runtime_error("output write failed");
            RT(cudaFree(pointers[i]));
        }
        // Record dynamic libraries to independently audit the CPU-only backend.
        std::ifstream maps("/proc/self/maps");
        std::ofstream saved_maps("loaded-libraries.txt");
        saved_maps << maps.rdbuf();
        std::cout << "REPLAY_OUTPUTS_WRITTEN\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "REPLAY_ERROR: " << error.what() << '\n';
        return 2;
    }
}
