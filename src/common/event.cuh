#pragma once

#include <cuda_runtime.h>
#include "error.cuh"

class CudaEvent
{
private:
    cudaEvent_t event;

public:
    CudaEvent() : event(nullptr) { CUDA_CHECK(cudaEventCreate(&event)); }

    ~CudaEvent()
    {
        if (event) {
            cudaEventDestroy(event);
        }
    }

    // Disable copy semantics to prevent double-free corruption
    CudaEvent(const CudaEvent &other) = delete;
    CudaEvent &operator=(const CudaEvent &other) = delete;

    // Allow move semantics for efficient ownership transfer
    CudaEvent(CudaEvent &&other) noexcept : event(other.event)
    {
        other.event = nullptr;
    }

    CudaEvent &operator=(CudaEvent &&other) noexcept
    {
        if (this != &other) {
            if (event) {
                cudaEventDestroy(event);
            }
            event = other.event;
            other.event = nullptr;
        }
        return *this;
    }

    void record(cudaStream_t stream = 0)
    {
        CUDA_CHECK(cudaEventRecord(event, stream));
    }

    void synchronize() { CUDA_CHECK(cudaEventSynchronize(event)); }

    cudaEvent_t get() const { return event; }

    static float elapsed_time(CudaEvent &start, CudaEvent &stop)
    {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
        return ms;
    }
};
