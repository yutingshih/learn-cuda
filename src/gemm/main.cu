#include <fstream>
#include <iostream>
#include <utility>

#include "bench.cuh"
#include "kernels/gemm_coalescing.cuh"
#include "kernels/gemm_cpu.hpp"
#include "kernels/gemm_naive.cuh"
#include "kernels/gemm_shared_mem.cuh"
#include "kernels/sgemm_cublas.cuh"

void benchmarking(const int M,
                  const int N,
                  const int K,
                  const int iter = 10,
                  const std::string &filename = "result.csv")
{
    static std::pair<std::string, gemm_fn<float>> func[] = {
        {"Naive", gemm_naive<float>},
        {"Coalesced", gemm_coalescing<float>},
        {"SharedMem", gemm_shared_mem<float>},
        {"cuBLAS", sgemm_cublas},
    };

    std::printf("GEMM Optimization Benchmark: M=%d, N=%d, K=%d\n", M, N, K);
    BenchmarkResult res;
    std::ofstream file(filename);
    file << "name,time(ms),perf(GFLOPS)" << std::endl;

    for (const auto &fn : func) {
        bench_cuda<float>(fn.second, gemm_cpu, M, N, K, iter, &res);
        res.name = fn.first;
        std::cout << res << std::endl;
        file << fn.first << "," << res.time_ms << "," << res.gflops << "\n";
    }

    file.close();
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
