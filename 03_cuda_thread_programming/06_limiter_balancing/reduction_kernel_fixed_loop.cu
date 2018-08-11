#include <stdio.h>
#include "reduction.h"

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/
__global__ void
reduction_kernel(float *g_out, float *g_in, unsigned int size, int n_loads)
{
    unsigned int idx_x = blockIdx.x * n_loads * blockDim.x + threadIdx.x;

    extern __shared__ float s_data[];

    // Stride over grid and add the values to a shared memory buffer
    #if 0
    float input = 0.f;
    for (int i = 0; i < n_loads; i++) {
        input += ((idx_x + i*blockDim.x) < size) ? g_in[idx_x + i*blockDim.x] : 0.f;
    }
    s_data[threadIdx.x] = input;
    #else
    float input = 0.f;
    for (int i = idx_x; i < size; i += blockDim.x * gridDim.x) {
        input += g_in[i];
    }
    s_data[threadIdx.x] = input;
    #endif

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
    
    int n_loads = 1;
    
    #if 0
    int block_size = n_loads * n_threads;
    int n_blocks = (size + block_size - 1) / block_size;
    reduction_kernel<<<n_blocks, n_threads, n_threads * sizeof(float), 0>>>(g_outPtr, g_inPtr, size, n_loads);
    #else

    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_kernel, n_threads, n_threads*sizeof(float));
    //int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);
    reduction_kernel<<<n_blocks, n_threads, n_threads * sizeof(float), 0>>>(g_outPtr, g_inPtr, size, n_loads);

    printf("n_blocks:%d\n", n_blocks);

    size = n_blocks;
    n_blocks = 1;
    reduction_kernel<<<n_blocks, n_threads, n_threads * sizeof(float), 0>>>(g_outPtr, g_inPtr, size, n_loads);
    #endif

    printf("n_blocks:%d\n", n_blocks);
 
    return n_blocks;
}