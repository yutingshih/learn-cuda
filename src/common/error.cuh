#pragma once

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

inline void cuda_check(cudaError_t err, const char *file, int line)
{
    if (err == cudaSuccess)
        return;
    std::stringstream ss;
    ss << "[CUDA Error] " << file << ":" << line << " -> "
       << cudaGetErrorString(err);
    throw std::runtime_error(ss.str());
}

#define CUDA_CHECK(expr) cuda_check((expr), __FILE__, __LINE__)
