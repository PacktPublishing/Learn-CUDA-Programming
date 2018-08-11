#include <stdio.h>
#include <stdlib.h>

__global__ void
naive_reduction_kernel(volatile float *data_out, float *data_in, int stride, int size)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_x + stride < size) {
        data_out[idx_x] += data_in[idx_x + stride];
    }
}

void naive_reduction(float *d_out, float *d_in, int n_threads, int size)
{
    int n_blocks = (size + n_threads - 1) / n_threads;
    for (int stride = 1; stride < size; stride *= 2) {
        naive_reduction_kernel<<<n_blocks, n_threads>>>(d_out, d_in, stride, size);
    }
}

__global__ void
atomic_reduction_kernel(float *data_out, float *data_in, int size)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    atomicAdd(&data_out[0], data_in[idx_x]);
}

void atomic_reduction(float *d_out, float *d_in, int n_threads, int size)
{
    int n_blocks = (size + n_threads - 1) / n_threads;
    atomic_reduction_kernel<<<n_blocks, n_threads>>>(d_out, d_in, size);
}

