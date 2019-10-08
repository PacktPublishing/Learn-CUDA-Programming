#include <stdio.h>
#include <cooperative_groups.h>
#include "reduction.h"

using namespace cooperative_groups;

#define NUM_LOAD 4

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

// cuda thread synchronization
__global__ void
reduction_kernel(float* g_out, float* g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    thread_block block = this_thread_block();

    extern __shared__ float s_data[];

    // cumulates input with grid-stride loop and save to share memory
    float input[NUM_LOAD] = {0.f};
    for (int i = idx_x; i < size; i += block.group_dim().x * gridDim.x * NUM_LOAD)
    {
        for (int step = 0; step < NUM_LOAD; step++)
            input[step] += (i + step * block.group_dim().x * gridDim.x < size) ? g_in[i + step * block.group_dim().x * gridDim.x] : 0.f;
    }
    for (int i = 1; i < NUM_LOAD; i++)
        input[0] += input[i];
    s_data[threadIdx.x] = input[0];

    block.sync();

    // do reduction
    for (unsigned int stride = block.group_dim().x / 2; stride > 0; stride >>= 1) {
        // scheduled threads reduce for every iteration
        // and will be smaller than a warp size (32) eventually.
        if (block.thread_index().x < stride) { 
            s_data[block.thread_index().x] += s_data[block.thread_index().x + stride];

            // __syncthreads(); // Step 3: Error
            // block.sync();    // Step 4: Benefit of cooperative group, performance may drop but provides programming flexibility
        }
        // __syncthreads(); // Step 1: Original
        block.sync();       // Step 2: Equivalent operation
    }

    if (block.thread_index().x == 0) {
        g_out[block.group_index().x] = s_data[0];
    }
}

void reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_kernel, n_threads, n_threads*sizeof(float));
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    reduction_kernel<<< n_blocks, n_threads, n_threads * sizeof(float), 0 >>>(g_outPtr, g_inPtr, size);
    reduction_kernel<<< 1, n_threads, n_threads * sizeof(float), 0 >>>(g_outPtr, g_inPtr, n_blocks);
}
