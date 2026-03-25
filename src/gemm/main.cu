#include <fstream>
#include <iostream>

#include "bench.cuh"
#include "kernels/gemm_coalescing.cuh"
#include "kernels/gemm_cpu.hpp"
#include "kernels/gemm_naive.cuh"
#include "kernels/sgemm_cublas.cuh"

#define FUNC_NAME_WIDTH 14

static inline void print_result(const std::string &name,
                                float time_ms,
                                float gflops,
                                bool passed)
{
    std::printf("[%s] time = %f ms, perf = %f GFLOPS => %s\n", name.c_str(),
                time_ms, gflops, passed ? "Pass" : "Fail");
}

void benchmarking(const int M,
                  const int N,
                  const int K,
                  const int iter = 10,
                  const std::string &filename = "result.csv")
{
    std::printf("GEMM Optimization Benchmark: M=%d, N=%d, K=%d\n", M, N, K);
    float time_ms = 0.0, gflops = 0.0;
    bool passed = false;
    std::ofstream out(filename);
    out << "name,time(ms),perf(GFLOPS)" << std::endl;

    bench_cuda<float>(gemm_naive<float>, gemm_cpu, M, N, K, iter, &passed,
                      &time_ms, &gflops);
    print_result("Naive", time_ms, gflops, passed);
    out << "Naive" << "," << time_ms << "," << gflops << std::endl;

    bench_cuda<float>(gemm_coalescing<float>, gemm_cpu, M, N, K, iter, &passed,
                      &time_ms, &gflops);
    print_result("Coalesced", time_ms, gflops, passed);
    out << "Coalesced" << "," << time_ms << "," << gflops << std::endl;

    bench_cuda<float>(sgemm_cublas, gemm_cpu, M, N, K, iter, &passed, &time_ms,
                      &gflops);
    print_result("cuBLAS", time_ms, gflops, passed);
    out << "cuBLAS" << "," << time_ms << "," << gflops << std::endl;

    out.close();
}

int main()
{
    std::cout << "========[base]========" << std::endl;
    benchmarking(1024, 1024, 1024, 10, "data/base_1024_1024_1024.csv");

    std::cout << "========[qkv]========" << std::endl;
    benchmarking(197, 1152, 384, 10, "data/qkv_197_1152_384.csv");

    std::cout << "========[out]========" << std::endl;
    benchmarking(197, 384, 384, 10, "data/out_197_384_384.csv");

    std::cout << "========[fc1]========" << std::endl;
    benchmarking(197, 1536, 384, 10, "data/fc1_197_1536_384.csv");

    std::cout << "========[fc2]========" << std::endl;
    benchmarking(197, 384, 1536, 10, "data/fc2_197_384_1536.csv");

    return 0;
}
