# Learn CUDA

A cellection of CUDA code examples and notes to learn CUDA programming.

## Prerequisites

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx)
- [GNU Compiler Collection (GCC)](https://gcc.gnu.org/)
- [GNU Make](https://www.gnu.org/software/make/) or [Ninja](https://ninja-build.org/)
- [CMake](https://cmake.org/)
- [OpenCV](https://opencv.org/)

## Getting Started

Start and run a Docker container with neccessary libraries and tools:

```shell
./scripts/docker.sh build
./scripts/docker.sh run
```

Compile the source code:

```shell
cmake -B build -G Ninja
cmake --build build
```

Executable the binary or profile with [Nsight Systems](https://developer.nvidia.com/nsight-systems)

```shell
# execute directly
./build/vecadd

# profile with Nsight Systems
nsys -o report/vecadd ./build/vecadd
```

You may need to download [Nsight Systems Host](https://developer.nvidia.com/nsight-systems/get-started) (Windows/Linux/macOS are available) on your own computer to visualize the profiling report.

## Additional Resources

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Programming Massively Parallel Processors: A Hands-on Approach, 3rd Edition (PMPP3)](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0128119861)
- [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
