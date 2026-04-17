#pragma once

#include <cuda_runtime.h>

#include "error.cuh"

template <typename T>
class DeviceBuffer
{
private:
    T *ptr;
    size_t numel;

public:
    DeviceBuffer(size_t numel) : numel(numel), ptr(nullptr)
    {
        CUDA_CHECK(cudaMalloc(&ptr, numel * sizeof(T)));
    }

    ~DeviceBuffer() { cudaFree(ptr); }

    // Disable copy semantics to prevent double-free corruption
    DeviceBuffer(const DeviceBuffer &other) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &other) = delete;

    // Allow move semantics for efficient ownership transfer
    DeviceBuffer(DeviceBuffer &&other) noexcept
        : ptr(other.ptr), numel(other.numel)
    {
        other.ptr = nullptr;
        other.numel = 0;
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this != &other) {
            cudaFree(ptr);
            ptr = other.ptr;
            numel = other.numel;
            other.ptr = nullptr;
            other.numel = 0;
        }
        return *this;
    }

    T *data() const { return ptr; }
    size_t size() const { return numel; }
    size_t nbytes() const { return numel * sizeof(T); }
};
