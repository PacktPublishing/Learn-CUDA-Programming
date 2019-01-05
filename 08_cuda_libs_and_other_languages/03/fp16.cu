#include "fp16.cuh"
#include <cuda_fp16.h>

#define BLOCK_DIM 512

namespace fp16
{
__global__ void float2half_kernel(half *out, float *in)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    out[idx] = __float2half(in[idx]);
}

__global__ void half2float_kernel(float *out, half *in)
{ 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    out[idx] = __half2float(in[idx]);
}

void float2half(half *out, float *in, size_t length)
{
    float2half_kernel<<< (length + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM >>>(out, in);
}

void half2float(float *out, half *in, size_t length)
{
    half2float_kernel<<< (length + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM >>>(out, in);
}
} // namespace fp16