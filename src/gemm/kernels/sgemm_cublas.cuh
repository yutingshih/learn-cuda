#ifndef SGEMM_CUBLAS_CUH
#define SGEMM_CUBLAS_CUH

#include <cublas_v2.h>

void sgemm_cublas(const float *a,
                  const float *b,
                  float *c,
                  int m,
                  int n,
                  int k,
                  float alpha,
                  float beta)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k,
                &beta, c, n);
    cublasDestroy(handle);
}

#endif  // SGEMM_CUBLAS_CUH
