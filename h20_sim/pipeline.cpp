// CPU simulator replay protocol v1. Allocations persist across every launch.
// No tensor mathematics or reference data are implemented in this harness.
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void rt(cudaError_t s, const char* call) {
    if (s != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + std::to_string(s));
}
static void drv(CUresult s, const char* call) {
    if (s != CUDA_SUCCESS) throw std::runtime_error(std::string(call) + ": " + std::to_string(s));
}
#define RT(call) rt((call), #call)
#define DRV(call) drv((call), #call)
constexpr size_t guard_size = 128;
constexpr size_t memory_limit = 1024ull * 1024 * 1024;
struct Buffer {
    std::string name;
    size_t size;
    bool readonly;
    void* allocation = nullptr;
    std::vector<char> initial;
};
static std::vector<char> read_file(const std::string& name, size_t expected) {
    std::ifstream f(name, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != static_cast<std::streamoff>(expected))
        throw std::runtime_error("input size mismatch: " + name);
    std::vector<char> bytes(expected);
    f.seekg(0);
    if (expected && !f.read(bytes.data(), expected)) throw std::runtime_error("input read: " + name);
    return bytes;
}
static void save(const std::string& name, const std::vector<char>& bytes) {
    std::ofstream f(name, std::ios::binary);
    if (!f.write(bytes.data(), bytes.size())) throw std::runtime_error("output write: " + name);
}
static void snapshot(const std::vector<Buffer>& buffers, unsigned step) {
    for (const auto& b : buffers) {
        std::vector<char> bytes(b.size + 2 * guard_size);
        RT(cudaMemcpy(bytes.data(), b.allocation, bytes.size(), cudaMemcpyDeviceToHost));
        std::ostringstream name;
        name << "step" << std::setw(3) << std::setfill('0') << step << "-" << b.name << ".bin";
        save(name.str(), bytes); // retain evidence even if a guard is broken
        for (size_t i = 0; i < guard_size; ++i)
            if (static_cast<unsigned char>(bytes[i]) != 0xa5 ||
                static_cast<unsigned char>(bytes[guard_size + b.size + i]) != 0xa5)
                throw std::runtime_error("buffer guard damaged: " + b.name);
        if (b.readonly && bytes != b.initial)
            throw std::runtime_error("read-only input modified: " + b.name);
    }
}
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: pipeline manifest expected_simulator_library");
        Dl_info info{};
        if (!dladdr(reinterpret_cast<void*>(cuLaunchKernel), &info) || !info.dli_fname ||
            std::string(info.dli_fname) != argv[2])
            throw std::runtime_error("unexpected CUDA library");
        std::ifstream manifest(argv[1]);
        std::string magic;
        size_t buffer_count, launch_count;
        manifest >> magic >> buffer_count >> launch_count;
        if (magic != "SIM002_V1" || buffer_count > 256 || launch_count > 128)
            throw std::runtime_error("invalid replay manifest");
        RT(cudaSetDevice(0));
        std::vector<Buffer> buffers;
        size_t total = 0;
        for (size_t i = 0; i < buffer_count; ++i) {
            Buffer b;
            std::string file;
            manifest >> std::quoted(b.name) >> b.size >> b.readonly >> std::quoted(file);
            if (!manifest || b.size > memory_limit || total + b.size + 2 * guard_size > memory_limit)
                throw std::runtime_error("invalid or over-budget buffer");
            total += b.size + 2 * guard_size;
            b.initial = read_file(file, b.size + 2 * guard_size);
            RT(cudaMalloc(&b.allocation, b.initial.size()));
            RT(cudaMemcpy(b.allocation, b.initial.data(), b.initial.size(), cudaMemcpyHostToDevice));
            buffers.push_back(std::move(b));
        }
        std::map<std::string, CUmodule> modules;
        for (size_t step = 0; step < launch_count; ++step) {
            std::string file, entry;
            unsigned gx, gy, gz, threads, shared, argc_kernel;
            manifest >> std::quoted(file) >> std::quoted(entry) >> gx >> gy >> gz >> threads >> shared >> argc_kernel;
            if (!manifest || !gx || !gy || !gz || !threads || threads > 1024 || argc_kernel > 128)
                throw std::runtime_error("invalid launch");
            std::vector<uint64_t> storage(argc_kernel);
            for (unsigned j = 0; j < argc_kernel; ++j) {
                char kind;
                manifest >> kind;
                if (kind == 'P') {
                    size_t index, offset;
                    manifest >> index >> offset;
                    if (index >= buffers.size() || offset > buffers[index].size)
                        throw std::runtime_error("invalid buffer reference");
                    storage[j] = reinterpret_cast<uintptr_t>(buffers[index].allocation) + guard_size + offset;
                } else if (kind == 'S') {
                    manifest >> storage[j];
                } else throw std::runtime_error("unknown argument kind");
            }
            if (!manifest) throw std::runtime_error("truncated arguments");
            if (!modules.count(file)) {
                CUmodule module;
                DRV(cuModuleLoad(&module, file.c_str()));
                modules.emplace(file, module);
            }
            CUfunction kernel;
            DRV(cuModuleGetFunction(&kernel, modules.at(file), entry.c_str()));
            if (shared > 48 * 1024)
                DRV(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared));
            std::vector<void*> args;
            for (auto& value : storage) args.push_back(&value);
            std::cout << "SIM002_LAUNCH " << step << " " << entry << std::endl;
            DRV(cuLaunchKernel(kernel, gx, gy, gz, threads, 1, 1, shared, nullptr, args.data(), nullptr));
            RT(cudaDeviceSynchronize());
            snapshot(buffers, step);
            std::cout << "SIM002_STEP_COMPLETE " << step << std::endl;
        }
        std::string extra;
        if (manifest >> extra) throw std::runtime_error("trailing manifest data");
        std::ifstream maps("/proc/self/maps");
        std::ofstream saved("loaded-libraries.txt");
        saved << maps.rdbuf();
        for (auto& b : buffers) RT(cudaFree(b.allocation));
        std::cout << "SIM002_OUTPUTS_WRITTEN " << launch_count << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "SIM002_REPLAY_ERROR: " << e.what() << std::endl;
        return 2;
    }
}
