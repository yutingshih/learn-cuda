#ifndef MATMUL_H
#define MATMUL_H

#include <stdio.h>
#include <stdlib.h>

void matInit(float *M, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            M[i * size + j] = rand() % 10;
        }
    }
}

void matShow(float *M, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%6.1f ", M[i * size + j]);
        }
        puts("");
    }
    puts("");
}

#endif // MATMUL_H
