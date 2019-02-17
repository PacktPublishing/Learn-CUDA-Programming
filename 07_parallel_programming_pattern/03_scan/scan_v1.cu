#include "scan.h"

__global__ void
scan_v1_kernel(float *d_output, float *d_input, int length, int offset)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= length)
        return;

    if (idx - offset >= 0)
        d_output[idx] += d_input[idx - offset];
}

void scan_v1(float *d_output, float *d_input, int length)
{
    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid((length + BLOCK_DIM - 1) / BLOCK_DIM);
    for (int offset = 0; offset < length; offset++)
    {
        scan_v1_kernel<<<dimGrid, dimBlock>>>(d_output, d_input, length, offset);
        cudaDeviceSynchronize();
    }
}