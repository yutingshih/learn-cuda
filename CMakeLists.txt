cmake_minimum_required(VERSION 3.10)
project(learn-cuda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)
find_package(CUDAToolkit REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hello src/examples/hello/hello.cu)
add_executable(vecadd src/examples/vecadd/vecAdd.cu)
add_executable(random src/examples/random/random.cu)
add_executable(matmul.cpu src/examples/matmul/matmul.c)
add_executable(matmul src/examples/matmul/matmul.cu)
add_executable(conv2d
    src/examples/conv2d/main.cc
    src/examples/conv2d/conv.cu
)
target_include_directories(conv2d PRIVATE src/examples/conv2d)

add_executable(invert src/examples/invert_color/invert.cu)
target_link_libraries(invert ${OpenCV_LIBS})

add_executable(grayscale src/examples/grayscale/grayscale.cu)
target_link_libraries(grayscale ${OpenCV_LIBS})

add_executable(blur src/examples/blur/blur.cu)
target_link_libraries(blur ${OpenCV_LIBS})

add_executable(gemm src/gemm/main.cu)
target_include_directories(gemm PRIVATE src/gemm)
target_include_directories(gemm PRIVATE src)
target_link_libraries(gemm PRIVATE CUDA::cublas)
