#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void _grayscale(uint8_t *data, size_t size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size / 3) {
        uint8_t r = data[i * 3];
        uint8_t g = data[i * 3 + 1];
        uint8_t b = data[i * 3 + 2];
        uint8_t gray = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
        data[i * 3] = data[i * 3 + 1] = data[i * 3 + 2] = gray;
    }
}

void grayscale(uint8_t *data, size_t size)
{
    uint8_t *_data = nullptr;
    cudaMalloc(&_data, size);
    cudaMemcpy(_data, data, size, cudaMemcpyDefault);
    size_t threads = 32;
    size_t blocks = 1 + (size - 1) / threads;
    _grayscale<<<blocks, threads>>>(_data, size);
    cudaMemcpy(data, _data, size, cudaMemcpyDefault);
    cudaFree(_data);
}

int main(int argc, const char *argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Could not read the image: " << argv[1] << std::endl;
        return -1;
    }

    uint8_t *data = img.ptr<uint8_t>();
    size_t size = img.total() * img.elemSize();
    grayscale(data, size);

    cv::imwrite("./grayscale.jpg", img);
    return 0;
}
