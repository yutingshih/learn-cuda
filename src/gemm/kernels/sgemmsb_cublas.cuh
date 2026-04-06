#ifndef SGEMMSB_CUBLAS_CUH
#define SGEMMSB_CUBLAS_CUH

#include <cublas_v2.h>

void sgemmsb_cublas(const float *matA,
                    const float *matB,
                    float *matC,
                    int B,
                    int M,
                    int N,
                    int K,
                    long long int strideA,
                    long long int strideB,
                    long long int strideC,
                    float alpha,
                    float beta)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                              matB, N, strideB, matA, K, strideA, &beta, matC,
                              N, strideC, B);
    cublasDestroy(handle);
}

#endif  // SGEMMSB_CUBLAS_CUH
