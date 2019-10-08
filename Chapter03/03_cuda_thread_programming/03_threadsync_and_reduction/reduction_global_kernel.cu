#include <stdio.h>
#include <stdlib.h>

__global__ void
global_reduction_kernel(float *data_out, float *data_in, int stride, int size)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_x + stride < size) {
        data_out[idx_x] += data_in[idx_x + stride];
    }
}

void global_reduction(float *d_out, float *d_in, int n_threads, int size)
{
    int n_blocks = (size + n_threads - 1) / n_threads;
    for (int stride = 1; stride < size; stride *= 2) {
        global_reduction_kernel<<<n_blocks, n_threads>>>(d_out, d_in, stride, size);
    }
}
