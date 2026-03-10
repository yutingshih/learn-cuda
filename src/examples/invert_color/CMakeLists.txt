cmake_minimum_required(VERSION 3.10)
project(invert LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)
add_executable(invert invert.cu)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(invert ${OpenCV_LIBS})
