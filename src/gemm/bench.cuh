#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

template <typename T, typename AccT = T>
using gemm_fn = void (*)(const T *, const T *, T *, int, int, int, AccT, AccT);

template <typename T>
bool verify(const T *cpu_res, const T *gpu_res, int size, float tol = 1e-1)
{
    for (int i = 0; i < size; i++) {
        if (std::fabs(cpu_res[i] - gpu_res[i]) > tol) {
            std::cout << "Verification failed at index " << i << ": ";
            std::cout << "CPU=" << cpu_res[i] << ", ";
            std::cout << "GPU=" << cpu_res[i] << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T, typename AccT = T>
void bench_cpu(gemm_fn<T, AccT> gemm,
               const int m,
               const int n,
               const int k,
               const int iter,
               float *elapsed = nullptr,
               float *gflops = nullptr)
{
    std::vector<T> a(m * k, 1.1f);
    std::vector<T> b(k * n, 2.2f);
    std::vector<T> c(m * n, 0.0f);
    AccT alpha = 1.0f, beta = 0.0f;

    // Warm up
    gemm(a.data(), b.data(), c.data(), m, n, k, alpha, beta);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++) {
        gemm(a.data(), b.data(), c.data(), m, n, k, alpha, beta);
    }
    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = stop - start;
    if (elapsed && gflops) {
        *elapsed = duration.count() / iter;
        *gflops = (2.0 * m * n * k * 1e-9) / (*elapsed * 1e-3);
    }
}

template <typename T, typename AccT = T>
void bench_cuda(gemm_fn<T, AccT> gemm,
                gemm_fn<T, AccT> gemm_ref,
                const int m,
                const int n,
                const int k,
                const int iter,
                bool *passed = nullptr,
                float *elapsed = nullptr,
                float *gflops = nullptr)
{
    std::vector<T> a(m * k, 1.1f);
    std::vector<T> b(k * n, 2.2f);
    std::vector<T> c_host(m * n, 0.0f);
    std::vector<T> c_ref(m * n, 0.0f);
    AccT alpha = 1.0f, beta = 0.0f;

    gemm_ref(a.data(), b.data(), c_ref.data(), m, n, k, alpha, beta);

    T *_a, *_b, *_c;
    cudaMalloc(&_a, m * k * sizeof(T));
    cudaMalloc(&_b, k * n * sizeof(T));
    cudaMalloc(&_c, m * n * sizeof(T));

    cudaMemcpy(_a, a.data(), m * k * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b.data(), k * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(_c, 0, m * n * sizeof(T));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up
    gemm(_a, _b, _c, m, n, k, alpha, beta);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iter; i++) {
        gemm(_a, _b, _c, m, n, k, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    if (elapsed && gflops) {
        *elapsed = 0.0f;
        cudaEventElapsedTime(elapsed, start, stop);
        *elapsed /= iter;
        *gflops = (2.0 * m * n * k * 1e-9) / (*elapsed * 1e-3);
    }
    cudaMemcpy(c_host.data(), _c, m * n * sizeof(T), cudaMemcpyDeviceToHost);
    if (passed) {
        *passed = verify(c_ref.data(), c_host.data(), m * n);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(_a);
    cudaFree(_b);
    cudaFree(_c);
}
