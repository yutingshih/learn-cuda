#include <fstream>
#include <iostream>
#include <utility>

#include "bench.cuh"
#include "kernels/gemm_coalescing.cuh"
#include "kernels/gemm_cpu.hpp"
#include "kernels/gemm_naive.cuh"
#include "kernels/gemm_shared_mem.cuh"
#include "kernels/gemmsb_cpu.hpp"
#include "kernels/sgemm_cublas.cuh"
#include "kernels/sgemmsb_cublas.cuh"

void bench_gemm(const int M,
                const int N,
                const int K,
                const int iter = 10,
                const std::string &filename = "result.csv")
{
    static std::pair<std::string, gemm_fn<float>> func[] = {
        {"GEMM.Naive", gemm_naive<float>},
        {"GEMM.Coalesced", gemm_coalescing<float>},
        {"GEMM.SharedMem", gemm_shared_mem<float>},
        {"GEMM.cuBLAS", sgemm_cublas},
    };
    static std::pair<std::string, gemmsb_fn<float>> func2[] = {
        {"GEMMSB.cuBLAS", sgemmsb_cublas},
    };

    std::printf("GEMM Optimization Benchmark: M=%d, N=%d, K=%d\n", M, N, K);
    BenchmarkResult res;
    std::ofstream file(filename);
    file << "name,time(ms),perf(GFLOPS)" << std::endl;

    for (const auto &fn : func) {
        bench_cuda<float>(fn.second, M, N, K, iter, &res);
        res.name = fn.first;
        std::cout << res << std::endl;
        file << fn.first << "," << res.time_ms << "," << res.gflops << "\n";
    }

    for (const auto &fn : func2) {
        bench_cuda<float>(fn.second, 100, M, N, K, M * K, K * N, M * N, iter,
                          &res);
        res.name = fn.first;
        std::cout << res << std::endl;
        file << fn.first << "," << res.time_ms << "," << res.gflops << "\n";
    }

    auto fn = std::pair{"GEMM.CPU", gemm_cpu<float>};
    bench_cpu<float>(fn.second, M, N, K, iter, &res);
    res.name = fn.first;
    std::cout << res << std::endl;
    file << fn.first << "," << res.time_ms << "," << res.gflops << "\n";

    auto fn2 = std::pair{"GEMMSB.CPU", gemmsb_cpu<float>};
    bench_cpu<float>(fn2.second, 1, M, N, K, M * K, K * N, M * N, iter, &res);
    res.name = fn2.first;
    std::cout << res << std::endl;
    file << fn2.first << "," << res.time_ms << "," << res.gflops << "\n";
}

int main()
{
    std::cout << "========[base]========" << std::endl;
    bench_gemm(1024, 1024, 1024, 10, "data/base_1024_1024_1024.csv");

    std::cout << "========[qkv]========" << std::endl;
    bench_gemm(197, 1152, 384, 10, "data/qkv_197_1152_384.csv");

    std::cout << "========[out]========" << std::endl;
    bench_gemm(197, 384, 384, 10, "data/out_197_384_384.csv");

    std::cout << "========[fc1]========" << std::endl;
    bench_gemm(197, 1536, 384, 10, "data/fc1_197_1536_384.csv");

    std::cout << "========[fc2]========" << std::endl;
    bench_gemm(197, 384, 1536, 10, "data/fc2_197_384_1536.csv");

    return 0;
}
