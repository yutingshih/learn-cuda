#include <stdio.h>
#include <stdlib.h>

#include "matmul.h"

void matMul(float *M, float *N, float *P, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float sum = 0.0;
            for (int k = 0; k < size; ++k) {
                sum += M[i * size + k] * N[k * size + j];
            }
            P[i * size + j] = sum;
        }
    }
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
