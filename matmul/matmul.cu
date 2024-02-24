#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#include "matmul.h"

__global__
void _matMul(float *M, float *N, float *P, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= size || j >= size) return;

    float sum = 0.0;
    for (int k = 0; k < size; ++k) {
        sum += M[i * size + k] * N[k * size + j];
    }
    P[i * size + j] = sum;
}

void matMul(float *M, float *N, float *P, int size) {
    int nbytes = size * size * sizeof(float);
    float *_M, *_N, *_P;
    cudaMalloc(&_M, nbytes);
    cudaMalloc(&_N, nbytes);
    cudaMalloc(&_P, nbytes);

    cudaMemcpy(_M, M, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(_N, N, nbytes, cudaMemcpyHostToDevice);

    int tileSize = 2, tileNum = 1 + (size - 1) / tileSize;
    dim3 dimGrid(tileNum, tileNum, 1);
    dim3 dimBlock(tileSize, tileSize, 1);
    _matMul<<<dimGrid, dimBlock>>>(_M, _N, _P, size);

    cudaMemcpy(P, _P, nbytes, cudaMemcpyDeviceToHost);

    cudaFree(_M);
    cudaFree(_N);
    cudaFree(_P);
}

int main() {
    const int size = 4;
    int nbytes = size * size * sizeof(float);
    float *M = (float *)malloc(nbytes);
    float *N = (float *)malloc(nbytes);
    float *P = (float *)malloc(nbytes);

    matInit(M, size);
    matInit(N, size);
    matMul(M, N, P, size);
    matShow(M, size);
    matShow(N, size);
    matShow(P, size);

    free(M), free(N), free(P);
    return 0;
}
