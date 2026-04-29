#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "common/bench.cuh"
#include "common/buffer.cuh"
#include "kernels/gemm_coalescing.cuh"
#include "kernels/gemm_naive.cuh"
#include "kernels/gemm_shared_mem.cuh"
#include "kernels/sgemm_cublas.cuh"
#include "kernels/sgemmsb_cublas.cuh"

void bench_gemm(const int B,
                const int M,
                const int N,
                const int K,
                const int iters = 10,
                const std::string &filename = "result.csv")
{
    std::cout << "============================\n";
    std::printf("B=%d, M=%d, N=%d, K=%d\n", B, M, N, K);

    std::vector<float> a(B * M * K, 1.1f);
    std::vector<float> b(B * K * N, 2.2f);
    std::vector<float> c(B * M * N, 0.0f);
    float alpha = 1.0f, beta = 0.0f;

    DeviceBuffer<float> _a(B * M * K);
    DeviceBuffer<float> _b(B * K * N);
    DeviceBuffer<float> _c(B * M * N);

    CUDA_CHECK(cudaMemcpy(_a.data(), a.data(), _a.nbytes(), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(_b.data(), b.data(), _b.nbytes(), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemset(_c.data(), 0, _c.nbytes()));

    uint64_t num_ops = 2ULL * B * M * N * K;
    BenchmarkResult res;

    res = bench("gemm.naive", iters, num_ops, [&]() {
        for (int i = 0; i < B; i++) {
            gemm_naive<float>(_a.data(), _b.data(), _c.data(), M, N, K, alpha,
                              beta);
        }
    });
    std::cout << res << std::endl;

    res = bench("gemm.coalesced", iters, num_ops, [&]() {
        for (int i = 0; i < B; i++) {
            gemm_coalescing<float>(_a.data(), _b.data(), _c.data(), M, N, K,
                                   alpha, beta);
        }
    });
    std::cout << res << std::endl;

    res = bench("gemm.shared_mem", iters, num_ops, [&]() {
        for (int i = 0; i < B; i++) {
            gemm_shared_mem<float>(_a.data(), _b.data(), _c.data(), M, N, K,
                                   alpha, beta);
        }
    });
    std::cout << res << std::endl;

    res = bench("gemm.cublas", iters, num_ops, [&]() {
        for (int i = 0; i < B; i++) {
            sgemm_cublas(_a.data(), _b.data(), _c.data(), M, N, K, alpha, beta);
        }
    });
    std::cout << res << std::endl;

    res = bench("gemmsb.cublas", iters, num_ops, sgemmsb_cublas, _a.data(),
                _b.data(), _c.data(), B, M, N, K, M * K, K * N, M * N, alpha,
                beta);
    std::cout << res << std::endl;


    CUDA_CHECK(cudaMemcpy(c.data(), _c.data(), _c.nbytes(), cudaMemcpyDefault));
    std::cout << "============================\n";
}

int main()
{
    int batch = 10;
    int iters = 10;

    bench_gemm(batch, 1024, 1024, 1024, iters, "data/tmp_1024_1024_1024.csv");
    bench_gemm(batch, 197, 1152, 384, iters, "data/qkv_197_1152_384.csv");
    bench_gemm(batch, 197, 384, 384, iters, "data/out_197_384_384.csv");
    bench_gemm(batch, 197, 1536, 384, iters, "data/fc1_197_1536_384.csv");
    bench_gemm(batch, 197, 384, 1536, iters, "data/fc2_197_384_1536.csv");

    return 0;
}
