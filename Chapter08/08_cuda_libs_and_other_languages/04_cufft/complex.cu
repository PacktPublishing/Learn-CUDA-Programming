#include <cufft.h>
#include "helper.cuh"

namespace op
{
__global__ void FloatToComplex_kernel(cufftComplex *complex, const float *real, const float *imag)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    complex[idx].x = real[idx];
    if (imag != nullptr)
        complex[idx].y = imag[idx];
}

void FloatToComplex(cufftComplex *complex, const float *real, const float *imag, const size_t length)
{
    dim3 dimBlock(512);
    dim3 dimGrid((length + dimBlock.x - 1) / dimBlock.x);

    FloatToComplex_kernel<<< dimGrid, dimBlock >>>(complex, real, imag);
}
}