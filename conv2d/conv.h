#ifndef CONV_H
#define CONV_H

/*
conv2d: 2 dimensional convolution
    I[N][C][H][W] * K[M][C][E][F] = O[N][M][R][S]

Arguments:
    I: input tensor
    K: kernel
    O: output tensor
    N: number of input tensors
    C: channel of input tensor
    H: height of input tensor
    W: width of input tensor
    M: number of kernels
    E: height of kernel
    F: width of kernel
    pad: padding of heights and widths
    str: strides of heights and widths
    dil: dilation rate of heights and widths
*/
void conv2d(double *I, double *K, double *O,
    unsigned N, unsigned C, unsigned H, unsigned W,
    unsigned M, unsigned E, unsigned F,
    unsigned pad, unsigned str, unsigned dil);

#endif // CONV_H
