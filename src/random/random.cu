#include <iostream>

#include <cuda.h>
#include <curand_kernel.h>

__global__ void randomNumber(float *x, int n) {
    curandState state;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        uint64_t seed = (uint64_t) clock() + (uint64_t) tid * 1234;
        curand_init(seed, tid, 0, &state);
        x[tid] = curand_uniform(&state);
    }
}

void printArray(float *a, int n) {
    for (int i = 0; i < n; i++)
        std::cout << a[i] << " ";
    std::cout << std::endl;
}

int main() {
    int nElem = 10;
    int nBytes = sizeof(float) * nElem;
    float *_x;
    cudaMalloc(&_x, nBytes);

    randomNumber<<<1, 10>>>(_x, nElem);

    float *x = (float *)malloc(nBytes);
    cudaMemcpy(x, _x, nBytes, cudaMemcpyDeviceToHost);
    printArray(x, nElem);
    cudaFree(_x);
    free(x);
    return 0;
}
