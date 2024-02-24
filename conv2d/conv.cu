#include "conv.h"

#include <iostream>
#include <cuda.h>

__global__
void _conv2d(double *I, double *K, double *O,
    unsigned N, unsigned C, unsigned H, unsigned W,
    unsigned M, unsigned E, unsigned F, unsigned R, unsigned S,
    unsigned pad, unsigned str, unsigned dil) {

    unsigned m = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned r = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned s = threadIdx.z + blockIdx.z * blockDim.z;
    if (m >= M || r >= R || s >= S) return;

    for (unsigned n = 0; n < N; ++n) {
        double psum = 0.0;
        for (unsigned c = 0; c < C; ++c)
        for (unsigned e = 0; e < E; ++e)
        for (unsigned f = 0; f < F; ++f) {
            psum += I[n*C*H*W + c*H*W + (r+e)*W + (s+f)] * K[m*C*E*F + c*E*F + e*F + f];
        }
        O[n*M*R*S + m*R*S + r*S + s] = psum;
    }
}

void conv2d(double *I, double *K, double *O,
    unsigned N, unsigned C, unsigned H, unsigned W,
    unsigned M, unsigned E, unsigned F,
    unsigned pad, unsigned str, unsigned dil) {
    unsigned R = (H + pad + pad - E) / str + 1;
    unsigned S = (W + pad + pad - F) / str + 1;

    double *_I, *_K, *_O;
    cudaMalloc((void **)&_I, N*C*H*W*sizeof(double));
    cudaMalloc((void **)&_K, M*C*E*F*sizeof(double));
    cudaMalloc((void **)&_O, N*M*R*S*sizeof(double));

    cudaMemcpy(_I, I, N*C*H*W*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(_K, K, M*C*E*F*sizeof(double), cudaMemcpyHostToDevice);
    unsigned tileNum = 4;
    dim3 gridSize(1, tileNum, tileNum);
    dim3 blockSize(M, R/tileNum, S/tileNum);
    _conv2d<<<gridSize, blockSize>>>(_I, _K, _O, N, C, H, W, M, E, F, R, S, pad, str, dil);
    cudaMemcpy(O, _O, N*M*R*S*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(_I);
    cudaFree(_K);
    cudaFree(_O);
}
