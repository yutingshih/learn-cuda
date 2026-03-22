#ifndef GEMM_NAIVE_CUH
#define GEMM_NAIVE_CUH

#include <cuda_runtime.h>

template <typename T, typename AccT = T>
__global__ void _gemm_naive(const T *a,
                            const T *b,
                            T *c,
                            int m,
                            int n,
                            int k,
                            AccT alpha,
                            AccT beta)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        AccT sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += static_cast<AccT>(a[row * k + i]) *
                   static_cast<AccT>(b[i * n + col]);
        }
        c[row * n + col] =
            static_cast<T>(alpha * sum + beta * c[row * n + col]);
    }
}

template <typename T, typename AccT = T>
void gemm_naive(const T *a,
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
    _gemm_naive<T, AccT>
        <<<grid_size, block_size>>>(a, b, c, m, n, k, alpha, beta);
}

#endif  // GEMM_NAIVE_CUH
