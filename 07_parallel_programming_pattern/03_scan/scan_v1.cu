#include "scan.h"

__global__ void
scan_v1_kernel(float *d_output, float *d_input, int length)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
        
    float element = 0.f;
    for (int offset = 0; offset < length; offset++) {
        if (idx - offset >= 0)
            element += d_input[idx - offset];
    }
    d_output[idx] = element;
}

void scan_v1(float *d_output, float *d_input, int length)
{
    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid((length + BLOCK_DIM - 1) / BLOCK_DIM);
    scan_v1_kernel<<<dimGrid, dimBlock>>>(d_output, d_input, length);
}