#pragma once

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "error.cuh"
#include "event.cuh"

struct BenchmarkResult {
    std::string name;
    float time_ms;
    uint64_t num_ops;

    double gflops() const
    {
        if (num_ops == 0 || time_ms == 0)
            return 0.0;
        return static_cast<double>(num_ops) / (time_ms * 1e6);
    }
};

std::ostream &operator<<(std::ostream &os, BenchmarkResult &res)
{
    os << std::left << std::setw(20) << res.name;
    os << std::fixed << std::setprecision(4) << std::right;
    os << " | Time: " << std::setw(10) << res.time_ms << " ms";
    if (res.num_ops > 0) {
        os << " | Perf: " << std::setw(10) << res.gflops() << " GFLOPS";
    }
    return os;
}

template <typename Func, typename... Args>
auto bench(const std::string &name,
           uint32_t iters,
           uint64_t num_ops,
           Func &&kernel,
           Args &&...args) -> BenchmarkResult
{
    // Warm up
    kernel(args...);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmarking
    CudaEvent start, stop;
    start.record();
    for (uint32_t i = 0; i < iters; i++) {
        kernel(args...);
    }
    stop.record();
    stop.synchronize();

    BenchmarkResult res = {
        .name = name,
        .time_ms = CudaEvent::elapsed_time(start, stop) / iters,
        .num_ops = num_ops,
    };
    return res;
}

template <typename Func, typename... Args>
auto bench_cpu(const std::string &name,
               uint32_t iters,
               uint64_t num_ops,
               Func &&kernel,
               Args &&...args) -> BenchmarkResult
{
    // Warm up
    kernel(args...);

    // Benchmarking
    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < iters; i++) {
        kernel(args...);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;

    BenchmarkResult res = {
        .name = name,
        .time_ms = duration / iters,
        .num_ops = num_ops,
    };
    return res;
}

template <typename T>
bool verify(const T *res, const T *ref, int size, float tol = 1e-1)
{
    for (int i = 0; i < size; i++) {
        float diff = std::fabs(res[i] - ref[i]);
        float max_val = std::fmax(std::fabs(res[i]), std::fabs(ref[i]));
        if (max_val > 0.0f && (diff / max_val) > tol) {
            std::cout << "Verification failed at index " << i << ": ";
            std::cout << "res=" << res[i] << ", ";
            std::cout << "ref=" << ref[i] << std::endl;
            return false;
        }
    }
    return true;
}
