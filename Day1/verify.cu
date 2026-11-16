// Independent GPU checks. Build with nvcc -O3 -std=c++17 -arch=sm_120
// -x cu verify.cu -o verify, then run ./verify. --quick skips large inputs.
#include <cuda_runtime.h>
#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace problem_a {
#define solution solution_a
#include "1/code.cpp"
#undef solution
}
namespace problem_b {
#define solution solution_b
#ifndef ARGMAX_SOURCE
#define ARGMAX_SOURCE "2/code.cpp"
#endif
#include ARGMAX_SOURCE
#undef solution
}
namespace problem_c {
#define solution solution_c
#ifndef ARGMIN_SOURCE
#define ARGMIN_SOURCE "3/code.cpp"
#endif
#include ARGMIN_SOURCE
#undef solution
}

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::cerr << operation << ": " << cudaGetErrorString(result) << '\n';
        std::exit(1);
    }
}
#define CUDA(call) check((call), #call)

template <class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        CUDA(cudaMalloc(reinterpret_cast<void**>(&data), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    void upload(const std::vector<T>& values) {
        CUDA(cudaMemcpy(data, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download(size_t count) const {
        std::vector<T> result(count);
        CUDA(cudaMemcpy(result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }
};

template <class F> float time_ms(F call) {
    cudaEvent_t start, end;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&end));
    std::vector<float> samples;
    for (int repeat = 0; repeat < 5; ++repeat) {
        CUDA(cudaEventRecord(start));
        call();
        CUDA(cudaEventRecord(end));
        CUDA(cudaEventSynchronize(end));
        float ms;
        CUDA(cudaEventElapsedTime(&ms, start, end));
        samples.push_back(ms);
    }
    CUDA(cudaEventDestroy(start));
    CUDA(cudaEventDestroy(end));
    std::sort(samples.begin(), samples.end());
    return samples[2];
}

void graph_case(int n, int kind, bool timing) {
    std::mt19937 random(1116 + n + kind);
    std::vector<float> graph(static_cast<size_t>(n) * n, 0.0f);
    auto edge = [&](int i, int j, float weight) { graph[static_cast<size_t>(i) * n + j] = weight; };
    if (kind == 0) {
        // Random vertex order forces paths to cross tile boundaries. The three
        // components have no paths between them, even at the public sizes.
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), random);
        for (int group = 0; group < 3; ++group) {
            int begin = group * n / 3, size = (group + 1) * n / 3 - begin;
            for (int i = 0; i < size; ++i)
                for (int shift : {1, 7, 19})
                    edge(order[begin + i], order[begin + (i + shift) % size], 1 + random() % 255);
        }
    } else if (kind == 1) {
        for (float& value : graph) value = random() % 128;
    } else if (kind == 2) {
        edge(2, 66, 8388608.0f);
        edge(66, 130, 8388608.0f);
        edge(130, 132, 256.0f);
        edge(3, 67, 8388624.0f);
        edge(67, 131, 32.0f);
        edge(4, 68, 1073741824.0f);
        edge(68, 133, 1024.0f);
        edge(5, 69, 1.0f);
        edge(69, 134, 1.0f);
        edge(5, 134, 1073741824.0f);
    } else if (kind == 3) {
        for (int i = 0; i + 1 < n; ++i) edge(i, i + 1, 1.0f);
    }

    using Entry = std::pair<float, int>;
    std::vector<std::vector<Entry>> edges(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (graph[static_cast<size_t>(i) * n + j] > 0 && i != j)
                edges[i].emplace_back(graph[static_cast<size_t>(i) * n + j], j);

    // Independent all-pairs reference: Dijkstra, rather than blocked Floyd.
    std::vector<float> expected(static_cast<size_t>(n) * n, std::numeric_limits<float>::infinity());
    for (int source = 0; source < n; ++source) {
        float* distance = expected.data() + static_cast<size_t>(source) * n;
        distance[source] = 0;
        std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
        queue.emplace(0.0f, source);
        while (!queue.empty()) {
            const auto [cost, vertex] = queue.top();
            queue.pop();
            if (cost != distance[vertex]) continue;
            for (const auto& [weight, next] : edges[vertex]) {
                const float candidate = cost + weight;
                if (candidate < distance[next]) {
                    distance[next] = candidate;
                    queue.emplace(candidate, next);
                }
            }
        }
    }
    for (float& value : expected) if (std::isinf(value)) value = -1;

    const size_t count = graph.size();
    DeviceBuffer<float> input(count), output(count + 32);
    input.upload(graph);
    output.upload(std::vector<float>(count + 32, -12345.0f));
    auto call = [&] { problem_a::solution_a(input.data, output.data + 16, n); };
    call();
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
    const auto actual = output.download(count + 32);
    for (size_t i = 0; i < count + 32; ++i) {
        const float wanted = i < 16 || i >= count + 16 ? -12345.0f : expected[i - 16];
        if (actual[i] != wanted) {
            std::cerr << "FAIL A n=" << n << " kind=" << kind << " index=" << i
                      << " expected=" << wanted << " actual=" << actual[i] << '\n';
            std::exit(1);
        }
    }
    if (input.download(count) != graph) std::exit(2);
    std::cout << "PASS A n=" << n << " kind=" << kind;
    if (timing) std::cout << " median_ms=" << time_ms(call);
    std::cout << std::endl;
}

void reduction_case(const std::vector<int>& shape, int dim, int pattern, bool timing) {
    const int axis = dim < 0 ? dim + static_cast<int>(shape.size()) : dim;
    size_t count = 1, inner = 1, outer = 1;
    for (int size : shape) count *= size;
    for (int d = 0; d < axis; ++d) outer *= shape[d];
    for (size_t d = axis + 1; d < shape.size(); ++d) inner *= shape[d];
    const int length = shape[axis];
    const size_t outputs = outer * inner;
    std::vector<float> input(count);
    uint32_t state = 1116;
    for (size_t i = 0; i < count; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        float value = static_cast<int>(state % 1021) - 510;
        if (pattern == 1) value = 7.0f;
        if (pattern == 2) {
            value = static_cast<int>(state % 7) - 3;
            if (i % 67 == 0) value = std::numeric_limits<float>::infinity();
            if (i % 71 == 0) value = -std::numeric_limits<float>::infinity();
            if (i % 73 == 0) value = std::numeric_limits<float>::quiet_NaN();
            if (i % 79 == 0) value = -0.0f;
        }
        if (pattern == 3) value = std::numeric_limits<float>::infinity();
        if (pattern == 4) value = -std::numeric_limits<float>::infinity();
        if (pattern == 5) value = i % 2 == 0 ? -0.0f : 0.0f;
        if (pattern == 6) std::memcpy(&value, &state, sizeof(value));
        if (pattern == 7) {
            const float tiny = std::numeric_limits<float>::denorm_min();
            value = i % 7 == 0 ? tiny * 2 : i % 7 == 1 ? tiny : -tiny;
        }
        input[i] = value;
    }

    std::vector<int> maximum(outputs, 0), minimum(outputs, 0);
    // Traverse each row-major input slice in order. The first equal value wins.
    for (size_t o = 0; o < outer; ++o) {
        std::vector<float> high(inner), low(inner);
        for (int k = 0; k < length; ++k)
            for (size_t i = 0; i < inner; ++i) {
                const float value = input[(o * length + k) * inner + i];
                if (k == 0) {
                    high[i] = low[i] = value;
                } else {
                    if (!std::isnan(high[i]) && (std::isnan(value) || value > high[i])) {
                        high[i] = value;
                        maximum[o * inner + i] = k;
                    }
                    if (!std::isnan(low[i]) && (std::isnan(value) || value < low[i])) {
                        low[i] = value;
                        minimum[o * inner + i] = k;
                    }
                }
            }
    }

    // Check both aligned vector-load paths and the unaligned fallback.
    DeviceBuffer<float> device_input(count + 1);
    DeviceBuffer<int> device_shape(shape.size()), output(outputs + 32);
    device_shape.upload(shape);
    for (int input_offset : {0, 1}) {
      CUDA(cudaMemcpy(device_input.data + input_offset, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    for (int operation = 0; operation < 2; ++operation)
      for (int host_shape = 0; host_shape < 2; ++host_shape) {
        output.upload(std::vector<int>(outputs + 32, -12345));
        const int* shape_pointer = host_shape ? shape.data() : device_shape.data;
        auto call = [&] {
            if (operation == 0)
                problem_b::solution_b(device_input.data + input_offset, dim, output.data + 16,
                                      shape_pointer, shape.size());
            else
                problem_c::solution_c(device_input.data + input_offset, dim, output.data + 16,
                                      shape_pointer, shape.size());
        };
        call();
        CUDA(cudaGetLastError());
        CUDA(cudaDeviceSynchronize());
        const auto actual = output.download(outputs + 32);
        const auto& expected = operation == 0 ? maximum : minimum;
        for (size_t i = 0; i < outputs + 32; ++i) {
            const int wanted = i < 16 || i >= outputs + 16 ? -12345 : expected[i - 16];
            if (actual[i] != wanted) {
                std::cerr << "FAIL " << (operation == 0 ? 'B' : 'C') << " dim=" << dim
                          << " pattern=" << pattern << " index=" << i
                          << " expected=" << wanted << " actual=" << actual[i] << '\n';
                std::exit(1);
            }
        }
        std::cout << "PASS " << (operation == 0 ? 'B' : 'C') << " shape=";
        for (int size : shape) std::cout << size << ',';
        std::cout << " dim=" << dim << " pattern=" << pattern
                  << " shape_memory=" << (host_shape ? "CPU" : "GPU")
                  << " input_offset=" << input_offset;
        if (timing) std::cout << " median_ms=" << time_ms(call);
        std::cout << std::endl;
    }
    }
}

void platform_sample() {
    const std::vector<float> sample{
         1, -2,  3, -4,  5, -1.5f,  2.5f,  0,
        -3,  4, -1,  2, -5,  1.5f, -2.5f,  3.5f,
         2, -3,  4, -2,  1, -4,  3, -1,
        -4,  5, -3,  1, -2,  4, -1,  2,
         3, -4,  2, -1,  4, -3,  1, -2,
        -2,  3, -4,  5, -1,  2, -3,  4,
         4, -1,  2, -3,  5, -2,  3, -4,
        -1,  2, -3,  4, -2,  3, -4,  1
    };
    const int shape[]{8, 8};
    DeviceBuffer<float> input(sample.size());
    input.upload(sample);
    DeviceBuffer<int> output(8);
    problem_b::solution_b(input.data, 0, output.data, shape, 2);
    CUDA(cudaDeviceSynchronize());
    if (output.download(8) != std::vector<int>{6, 3, 2, 5, 0, 3, 2, 5}) std::exit(3);
    std::cout << "PASS B platform_sample shape=8,8 dim=0 shape_memory=CPU" << std::endl;
    problem_c::solution_c(input.data, 0, output.data, shape, 2);
    CUDA(cudaDeviceSynchronize());
    if (output.download(8) != std::vector<int>{3, 4, 5, 0, 1, 2, 7, 6}) std::exit(3);
    std::cout << "PASS C platform_sample shape=8,8 dim=0 shape_memory=CPU" << std::endl;
}

int main(int argc, char** argv) {
    const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
    const bool reductions_only = argc > 1 && std::string(argv[1]) == "--reductions";
    const bool specialized_only = argc > 1 && std::string(argv[1]) == "--specialized";
    cudaDeviceProp properties;
    CUDA(cudaGetDeviceProperties(&properties, 0));
    std::cout << "GPU=" << properties.name << " SM=" << properties.major << properties.minor
              << std::fixed << std::setprecision(6) << std::endl;
    if (specialized_only) {
        // Force every new strided specialization with ties, NaNs and infinities.
        for (int pattern : {1, 2, 3, 4, 5}) {
            reduction_case({32, 262144}, 0, pattern, false);
            reduction_case({257, 128, 128}, 1, pattern, false);
            reduction_case({256, 65536}, 0, pattern, false);
        }
        reduction_case({9, 64}, 1, 5, false);
        reduction_case({9, 1024}, 1, 5, false);
        for (int pattern : {6, 7}) {
            reduction_case({32, 262144}, 0, pattern, false);
            reduction_case({257, 128, 128}, 1, pattern, false);
            reduction_case({256, 65536}, 0, pattern, false);
            reduction_case({9, 64}, 1, pattern, false);
            reduction_case({9, 1024}, 1, pattern, false);
        }
        std::cout << "ALL SPECIALIZED CHECKS PASSED" << std::endl;
        return 0;
    }
    if (!reductions_only) {
        problem_a::solution_a(nullptr, nullptr, 0);
        for (int n : {1, 2, 31, 32, 33, 63, 64, 65, 127, 129, 257}) graph_case(n, 0, false);
        graph_case(137, 1, false);
        graph_case(193, 2, false);
        graph_case(257, 3, false);
        graph_case(65, 4, false);
        if (!quick)
            for (int n : {512, 1024, 2048, 4096}) graph_case(n, 0, true);
    }

    platform_sample();
    for (int length : {1, 7, 31, 32, 33, 64, 65, 1024})
        for (int pattern : {0, 1, 2, 3, 4}) reduction_case({9, length}, 1, pattern, false);
    for (int pattern : {0, 1, 2}) {
        reduction_case({7, 33, 5}, 1, pattern, false);
        reduction_case({2, 5, 1}, 1, pattern, false);
        reduction_case({3, 17, 16385}, 1, pattern, false);
        reduction_case({7, 5, 33}, -1, pattern, false);
        reduction_case({1}, 0, pattern, false);
    }
    reduction_case({0, 7, 3}, 1, 0, false);
    if (!quick) {
        reduction_case({16, 128, 256}, 1, 0, true);
        reduction_case({32, 512, 512}, 0, 0, true);
        reduction_case({8, 1024, 1024}, 2, 0, true);
        reduction_case({64, 128, 128, 128}, 2, 0, true);
        reduction_case({4, 256, 256, 256}, 1, 0, true);
        reduction_case({128, 64, 64, 64}, 3, 0, true);
    }
    std::cout << "ALL CHECKS PASSED" << std::endl;
}
