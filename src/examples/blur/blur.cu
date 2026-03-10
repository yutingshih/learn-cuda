#include <cuda_runtime.h>

#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void _blur(uint8_t* out, const uint8_t* in, size_t rows, size_t cols,
                      int ks) {
    int r = threadIdx.y + blockIdx.y * blockDim.y;
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= rows || c >= cols) return;

    for (int k = 0; k < 3; ++k) {
        int num = 0, val = 0;
        for (int i = r - ks / 2; i <= r + ks / 2; ++i) {
            for (int j = c - ks; j < c + ks; ++j) {
                if (i < 0 || i >= rows || j < 0 || j >= cols) continue;
                val += in[(i * cols + j) * 3 + k];
                num++;
            }
        }
        out[(r * cols + c) * 3 + k] = static_cast<uint8_t>(val / num);
    }
}

void blur(uint8_t* data, size_t rows, size_t cols, int ks = 3, int bs = 32) {
    uint8_t *_in, *_out;
    size_t size = rows * cols * 3 * sizeof(uint8_t);
    cudaMalloc(&_in, size);
    cudaMalloc(&_out, size);
    cudaMemcpy(_in, data, size, cudaMemcpyDefault);
    dim3 threads(bs, bs);
    dim3 blocks(1 + (cols - 1) / bs, 1 + (rows - 1) / bs);
    _blur<<<blocks, threads>>>(_out, _in, rows, cols, ks);
    cudaMemcpy(data, _out, size, cudaMemcpyDefault);
    cudaFree(_in);
    cudaFree(_out);
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

    unsigned char* data = img.ptr<unsigned char>();
    blur(data, img.rows, img.cols);

    cv::imwrite("blurred.jpg", img);
    return 0;
}
