#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "buffer.cuh"
#include "error.cuh"
#include "event.cuh"

template <typename T, typename AccT = T>
using gemm_fn = void (*)(const T *, const T *, T *, int, int, int, AccT, AccT);

template <typename T, typename AccT = T>
using gemmsb_fn = void (*)(const T *,
                           const T *,
                           T *,
                           int,
                           int,
                           int,
                           int,
                           long long int,
                           long long int,
                           long long int,
                           AccT,
                           AccT);

struct BenchmarkResult {
    std::string name;
    float time_ms;
    float gflops;
    uint64_t num_ops;
    bool passed;

    void set_result(float time_ms, uint64_t num_ops, bool passed = true)
    {
        this->time_ms = time_ms;
        this->num_ops = num_ops;
        this->gflops = (num_ops * 1e-9) / (time_ms * 1e-3);
        this->passed = passed;
    }
};

std::ostream &operator<<(std::ostream &os, const BenchmarkResult &res)
{
    os << "[" << res.name << "] ";
    os << "time = " << res.time_ms << " ms, ";
    os << "perf = " << res.gflops << " GFLOPS, ";
    os << "=> " << (res.passed ? "Passed" : "Failed");
    return os;
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

template <typename T, typename AccT = T>
void bench_cpu(gemm_fn<T, AccT> func,
               const int M,
               const int N,
               const int K,
               const int iter,
               BenchmarkResult *result = nullptr)
{
    std::vector<T> a(M * K, 1.1f);
    std::vector<T> b(K * N, 2.2f);
    std::vector<T> c(M * N, 0.0f);
    AccT alpha = 1.0f, beta = 0.0f;

    // Warm up
    func(a.data(), b.data(), c.data(), M, N, K, alpha, beta);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++) {
        func(a.data(), b.data(), c.data(), M, N, K, alpha, beta);
    }
    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = stop - start;
    if (result) {
        result->set_result(duration.count() / iter, 2.0 * M * N * K);
    }
}

template <typename T, typename AccT = T>
void bench_cpu(gemmsb_fn<T, AccT> func,
               const int B,
               const int M,
               const int N,
               const int K,
               const uint64_t str_a,
               const uint64_t str_b,
               const uint64_t str_c,
               const int iter,
               BenchmarkResult *result = nullptr)
{
    std::vector<T> a(B * M * K, 1.1f);
    std::vector<T> b(B * K * N, 2.2f);
    std::vector<T> c(B * M * N, 0.0f);
    AccT alpha = 1.0f, beta = 0.0f;

    // Warm up
    func(a.data(), b.data(), c.data(), B, M, N, K, str_a, str_b, str_c, alpha,
         beta);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++) {
        func(a.data(), b.data(), c.data(), B, M, N, K, str_a, str_b, str_c,
             alpha, beta);
    }
    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = stop - start;
    if (result) {
        result->set_result(duration.count() / iter, 2.0 * B * M * N * K);
    }
}

template <typename T, typename AccT = T>
void bench_cuda(gemm_fn<T, AccT> func,
                const int M,
                const int N,
                const int K,
                const int iter,
                BenchmarkResult *result = nullptr)
{
    std::vector<T> a(M * K, 1.1f);
    std::vector<T> b(K * N, 2.2f);
    std::vector<T> c(M * N, 0.0f);
    AccT alpha = 1.0f, beta = 0.0f;

    DeviceBuffer<T> _a(M * K);
    DeviceBuffer<T> _b(K * N);
    DeviceBuffer<T> _c(M * N);

    CUDA_CHECK(cudaMemcpy(_a.data(), a.data(), _a.nbytes(), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(_b.data(), b.data(), _b.nbytes(), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemset(_c.data(), 0, _c.nbytes()));

    CudaEvent start, stop;

    // Warm up
    func(_a.data(), _b.data(), _c.data(), M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    start.record();
    for (int i = 0; i < iter; i++) {
        func(_a.data(), _b.data(), _c.data(), M, N, K, alpha, beta);
    }
    stop.record();
    stop.synchronize();

    CUDA_CHECK(
        cudaMemcpy(c.data(), _c.data(), M * N * sizeof(T), cudaMemcpyDefault));

    if (result) {
        float time_ms = CudaEvent::elapsed_time(start, stop);
        result->set_result(time_ms / iter, 2.0 * M * N * K);
    }
}

template <typename T, typename AccT = T>
void bench_cuda(gemmsb_fn<T, AccT> func,
                const int B,
                const int M,
                const int N,
                const int K,
                const uint64_t str_a,
                const uint64_t str_b,
                const uint64_t str_c,
                const int iter,
                BenchmarkResult *result = nullptr)
{
    std::vector<T> a(B * M * K, 1.1f);
    std::vector<T> b(B * K * N, 2.2f);
    std::vector<T> c(B * M * N, 0.0f);
    AccT alpha = 1.0f, beta = 0.0f;

    DeviceBuffer<T> _a(B * M * K);
    DeviceBuffer<T> _b(B * K * N);
    DeviceBuffer<T> _c(B * M * N);

    CUDA_CHECK(cudaMemcpy(_a.data(), a.data(), _a.nbytes(), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(_b.data(), b.data(), _b.nbytes(), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemset(_c.data(), 0, _c.nbytes()));

    CudaEvent start, stop;

    // Warm up
    func(_a.data(), _b.data(), _c.data(), B, M, N, K, str_a, str_b, str_c,
         alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());

    start.record();
    for (int i = 0; i < iter; i++) {
        func(_a.data(), _b.data(), _c.data(), B, M, N, K, str_a, str_b, str_c,
             alpha, beta);
    }
    stop.record();
    stop.synchronize();

    CUDA_CHECK(cudaMemcpy(c.data(), _c.data(), B * M * N * sizeof(T),
                          cudaMemcpyDefault));

    if (result) {
        float time_ms = CudaEvent::elapsed_time(start, stop);
        result->set_result(time_ms / iter, 2.0 * B * M * N * K);
    }
}
