#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                               \
    do {                                               \
        cudaError_t err = (expr);                      \
        if (err != cudaSuccess) {                      \
            std::string msg = cudaGetErrorString(err); \
            throw std::runtime_error(msg);             \
        }                                              \
    } while (0)
