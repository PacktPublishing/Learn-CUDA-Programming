#include <stdio.h>
#include <cooperative_groups.h>
#include "reduction.h"

using namespace cooperative_groups;

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
    unsigned int idx_x = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

    thread_block block = this_thread_block();

    extern __shared__ float s_data[];

    s_data[block.thread_index().x] = (idx_x < size) ? g_in[idx_x] : 0.f;
    s_data[block.thread_index().x] += (idx_x + block.size() < size) ? g_in[idx_x + block.size()] : 0.f;

    block.sync();

    // do reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (block.thread_rank() < stride) { // scheduled threads reduce for every iteration, and will be smaller than a warp size (32) eventually.
            //coalesced_group active = coalesced_threads(); // Step 5: Warp scheduler selects CUDA threads which is 
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];

            // __syncthreads(); // Step 4: Error
            // block.sync();   // Step 3: Benefit of cooperative group, performance may drop but provides programming flexibility

            //active.sync(); // Step 5: Only required threads will do synchronization.
        }
        // __syncthreads(); // Step 1: Original
        block.sync(); // Step 2: Equivalent operation
    }

    if (block.thread_rank() == 0) {
        g_out[blockIdx.x] = s_data[0];
    }
}

int reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{   
    int block_size = 2 * n_threads;
    int n_blocks = (size + block_size - 1) / block_size;
    reduction_kernel<<< n_blocks, n_threads, n_threads * sizeof(float), 0 >>>(g_outPtr, g_inPtr, size);
    return n_blocks;
}
