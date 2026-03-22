#include "bench.cuh"
#include "kernels/gemm_cpu.hpp"
#include "kernels/gemm_naive.cuh"
#include "kernels/sgemm_cublas.cuh"

int main()
{
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const int iter = 10;

    std::printf("GEMM Optimization Benchmark: M=%d, N=%d, K=%d\n", M, N, K);

    bench_cuda<float>(gemm_naive<float>, gemm_cpu, M, N, K, iter, "Naive");
    bench_cuda<float>(sgemm_cublas, gemm_cpu, M, N, K, iter, "cuBLAS");

    return 0;
}
