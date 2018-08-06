#include <stdio.h>
#include "reduction.h"

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/
__global__ void
reduction_kernel(float *g_out, float *g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

    extern __shared__ float s_data[];

    // Stride over grid and add the values to a shared memory buffer    
    s_data[threadIdx.x] = (idx_x < size) ? g_in[idx_x] : 0.f;
    s_data[threadIdx.x] += ((idx_x + blockDim.x) < size) ? g_in[idx_x + blockDim.x] : 0.f;

    __syncthreads();

    // do reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride) 
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        g_out[blockIdx.x] = s_data[0];
    }
}

int reduction(float *g_outPtr, float *g_inPtr,
                 int size, int n_threads)
{
    int block_size = 2 * n_threads;
    int n_blocks = (size + block_size - 1) / block_size;
    reduction_kernel<<<n_blocks, n_threads, n_threads * sizeof(float), 0>>>(g_outPtr, g_inPtr, size);
 
    return n_blocks;
}