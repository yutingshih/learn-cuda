cmake_minimum_required(VERSION 3.10)
project(grayscale LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)
add_executable(grayscale grayscale.cu)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(grayscale ${OpenCV_LIBS})
