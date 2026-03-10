#include <cuda.h>

#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void _invert(uint8_t* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 255 - data[idx];
    }
}

void invert(uint8_t* data, size_t size) {
    uint8_t* _data;
    cudaMalloc(&_data, size);
    cudaMemcpy(_data, data, size, cudaMemcpyDefault);
    size_t block_size = 32;
    size_t grid_size = 1 + (size - 1) / block_size;
    _invert<<<grid_size, block_size>>>(_data, size);
    cudaMemcpy(data, _data, size, cudaMemcpyDefault);
    cudaFree(_data);
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Could not read the image: " << argv[1] << std::endl;
        return -1;
    }

    uint8_t* data = img.ptr<uint8_t>(0);
    size_t size = img.total() * img.elemSize();
    invert(data, size);

    cv::imwrite("./inverted.jpg", img);
    return 0;
}
