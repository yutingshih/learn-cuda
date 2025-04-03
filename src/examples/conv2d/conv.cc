#include "conv.h"

#include <iostream>

void conv2d(double *I, double *K, double *O,
    unsigned N, unsigned C, unsigned H, unsigned W,
    unsigned M, unsigned E, unsigned F,
    unsigned pad, unsigned str, unsigned dil) {
    unsigned R = (H + pad + pad - E) / str + 1;
    unsigned S = (W + pad + pad - F) / str + 1;

    for (unsigned n = 0; n < N; ++n)
    for (unsigned m = 0; m < M; ++m)
    for (unsigned r = 0; r < R; ++r)
    for (unsigned s = 0; s < S; ++s) {
        double psum = 0.0;
        for (unsigned c = 0; c < C; ++c)
        for (unsigned e = 0; e < E; ++e)
        for (unsigned f = 0; f < F; ++f) {
            psum += I[n*C*H*W + c*H*W + (r+e)*W + (s+f)] * K[m*C*E*F + c*E*F + e*F + f];
            #ifdef DEBUG
            std::printf("<DEBUG>: ");
            std::printf("I[%u, %u, %u, %u] * ", n, c, (r+e), (s+f));
            std::printf("K[%u, %u, %u, %u]  =  ", m, c, e, f);
            std::printf("%.2f * ", I[n*C*H*W + c*H*W + (r+e)*W + (s+f)]);
            std::printf("%.2f\n", K[m*C*E*F + c*E*F + e*F + f]);
            #endif
        }
        O[n*M*R*S + m*R*S + r*S + s] = psum;
    }
}
