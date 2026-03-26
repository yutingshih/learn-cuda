#ifndef GEMM_SHARED_MEM_CUH
#define GEMM_SHARED_MEM_CUH

#include <cuda_runtime.h>
#include <memory>

template <typename T,
          typename AccT = T,
          const int BM = 32,
          const int BN = 32,
          const int BK = 32>
__global__ void _gemm_shared_mem(const T *A,
                                 const T *B,
                                 T *C,
                                 int M,
                                 int N,
                                 int K,
                                 AccT alpha,
                                 AccT beta)
{
    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BM + ty;
    const int col = blockIdx.x * BK + tx;

    AccT sum{};

    for (int bk = 0; bk < K; bk += BK) {
        As[ty][tx] = A[row * K + (bk + tx)];
        Bs[ty][tx] = B[(bk + ty) * N + col];
        __syncthreads();

        for (int i = 0; i < BK; i++) {
            sum += static_cast<AccT>(As[ty][i]) * static_cast<AccT>(Bs[i][tx]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] =
            static_cast<T>(alpha * sum + beta * C[row * N + col]);
    }
}

template <typename T,
          typename AccT = T,
          const int BM = 32,
          const int BN = 32,
          const int BK = 32>
void gemm_shared_mem(const T *a,
                     const T *b,
                     T *c,
                     int m,
                     int n,
                     int k,
                     AccT alpha,
                     AccT beta)
{
    dim3 block_size(32, 32);
    dim3 grid_size(1 + (n - 1) / block_size.x, 1 + (m - 1) / block_size.y);
    _gemm_shared_mem<T, AccT, BM, BN, BK>
        <<<grid_size, block_size>>>(a, b, c, m, n, k, alpha, beta);
}

#endif  // GEMM_SHARED_MEM_CUH
