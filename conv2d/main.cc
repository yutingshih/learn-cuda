#include <iostream>
#include <iomanip>

#include "conv.h"

void printInfo(unsigned N, unsigned C, unsigned H, unsigned W,
    unsigned M, unsigned E, unsigned F, unsigned R, unsigned S,
    unsigned pad, unsigned str, unsigned dil) {
    std::printf(
        "<INFO> : I[%u][%u][%u][%u] * K[%u][%u][%u][%u] = O[%u][%u][%u][%u]  ",
        N, C, H, W, M, C, E, F, N, M, R, S
    );
    std::printf(
        "(padding = %u, stride = %u, dilation = %u)\n",
        pad, str, dil
    );
}

void printTensor(double *data, unsigned N, unsigned C, unsigned H, unsigned W, std::string sep="\t") {
    for (unsigned n = 0; n < N; ++n) {
        for (unsigned c = 0; c < C; ++c) {
            std::cout << "[" << n << ", " << c << "]" << std::endl;
            for (unsigned h = 0; h < H; ++h) {
                for (unsigned w = 0; w < W; ++w) {
                    unsigned idx = n*C*H*W + c*H*W + h*W + w;
                    std::cout << data[idx] << sep;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

void initTensor(double *data, unsigned size, int range = 2) {
    while (size) data[--size] = rand() % range;
}

void initTensor(double *data, unsigned N, unsigned C, unsigned H, unsigned W, int range = 2) {
    initTensor(data, N * C * H * W, range);
}

void testConv2d(double *imap, double *kern,
    unsigned N, unsigned C, unsigned H, unsigned W,
    unsigned M, unsigned E, unsigned F,
    unsigned pad, unsigned str, unsigned dil, bool verbose = true) {

    const unsigned R = (H + pad + pad - E) / str + 1;
    const unsigned S = (W + pad + pad - F) / str + 1;
    printInfo(N, C, H, W, M, E, F, R, S, pad, str, dil);

    double *_imap = imap;
    double *_kern = kern;
    double *_omap = (double *)malloc(N * M * R * S * sizeof(double));

    if (!imap) {
        _imap = (double *)malloc(N * C * H * W * sizeof(double));
        initTensor(_imap, N, C, H, W);
        #ifdef DEBUG
        std::cout << "<DEBUG>: " "_imap allocated" << std::endl;
        #endif
    }
    if (!kern) {
        _kern = (double *)malloc(M * C * E * F * sizeof(double));
        initTensor(_kern, N, C, E, F);
        #ifdef DEBUG
        std::cout << "<DEBUG>: " "_kern allocated" << std::endl;
        #endif
    }

    conv2d(_imap, _kern, _omap, N, C, H, W, M, E, F, pad, str, dil);

    if (verbose) {
        printTensor(_imap, N, C, H, W);
        printTensor(_kern, M, C, E, F);
        printTensor(_omap, N, M, R, S);
    }

    if (!imap) free(_imap);
    if (!kern) free(_kern);
    free(_omap);
}

void case1() {
    const unsigned pad = 0, str = 1, dil = 1;
    const unsigned N = 1, C = 1, H = 3, W = 3, M = 1, E = 3, F = 3;

    double *imap = (double *)malloc(H * W * sizeof(double));
    double *kern = (double *)malloc(E * F * sizeof(double));
    for (unsigned i = 0; i < H * W; ++i) imap[i] = 2.0;
    for (unsigned i = 0; i < E * F; ++i) kern[i] = 1.0;

    testConv2d(imap, kern, N, C, H, W, M, E, F, pad, str, dil);
    free(imap);
    free(kern);
}

void case2() {
    const unsigned pad = 0, str = 1, dil = 1;
    const unsigned N = 1, C = 1, H = 5, W = 5, M = 1, E = 3, F = 3;

    testConv2d(nullptr, nullptr, N, C, H, W, M, E, F, pad, str, dil);
}

void case3() {
    const unsigned pad = 1, str = 1, dil = 1;
    const unsigned N = 1, C = 2, H = 5, W = 5, M = 1, E = 3, F = 3;

    testConv2d(nullptr, nullptr, N, C, H, W, M, E, F, pad, str, dil);
}

void case4() {
    const unsigned pad = 0, str = 2, dil = 1;
    const unsigned N = 1, C = 2, H = 5, W = 5, M = 1, E = 3, F = 3;

    testConv2d(nullptr, nullptr, N, C, H, W, M, E, F, pad, str, dil);
}

void case5() {
    const unsigned pad = 1, str = 1, dil = 1;
    const unsigned N = 4, C = 3, H = 1024, W = 1024, M = 4, E = 5, F = 5;

    testConv2d(nullptr, nullptr, N, C, H, W, M, E, F, pad, str, dil, false);
}

void case6() {
    const unsigned pad = 1, str = 1, dil = 1;
    const unsigned N = 8, C = 8, H = 1024, W = 1024, M = 16, E = 3, F = 3;

    testConv2d(nullptr, nullptr, N, C, H, W, M, E, F, pad, str, dil, false);
}

int main() {
    case1();
    return 0;
}
