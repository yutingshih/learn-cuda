#include <cuda.h>
#include <stdio.h>

void vecInit(float *v, int n) {
    for (int i = 0; i < n; i++)
    v[i] = i + 1;
}

void vecPrint(float *v, int n) {
    for (int i = 0; i < n; i++)
    printf("%04.2f ", v[i]);
    puts("");
}

__global__ void _vecAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = b[i] + a[i];
}

void vecAdd(float *a, float *b, float *c, int n) {
    int size = n * sizeof(float);
    float *_a, *_b, *_c;
    cudaMalloc((void**)&_a, size);
    cudaMalloc((void**)&_b, size);
    cudaMalloc((void**)&_c, size);

    cudaMemcpy(_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, size, cudaMemcpyHostToDevice);

    _vecAdd<<<ceil(n/256.0), 256>>>(_a, _b, _c, n);

    cudaMemcpy(c, _c, size, cudaMemcpyDeviceToHost);

    cudaFree(_a);
    cudaFree(_b);
    cudaFree(_c);
}

int main() {
    float *a, *b, *c;
    int n = 20, size = n * sizeof(float);
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    vecInit(a, n);
    vecInit(b, n);

    vecPrint(a, n);
    vecPrint(b, n);
    vecAdd(a, b, c, n);
    vecPrint(c, n);

    free(a);
    free(b);
    free(c);
    return 0;
}
