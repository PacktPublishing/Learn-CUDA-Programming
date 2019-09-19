#include <stdio.h>
#include <cooperative_groups.h>
#include "reduction.h"

namespace cg = cooperative_groups;
using namespace cooperative_groups;

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

__device__ void
block_reduction(float *out, float *in, float *s_data, int active_size, int size, 
          const cg::grid_group &grid, const cg::thread_block &block)
{
    int tid = block.thread_rank();

    // Stride over grid and add the values to a shared memory buffer
    s_data[tid] = 0.f;
    for (int i = grid.thread_rank(); i < size; i += active_size)
        s_data[tid] += in[i];
    
    block.sync();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_data[tid] += s_data[tid + stride];
        
        block.sync();
    }

    if (block.thread_rank() == 0)
        out[block.group_index().x] = s_data[0];
}

__global__ void
reduction_kernel(float *g_out, float *g_in, unsigned int size)
{
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float s_data[];

    // do reduction for multiple blocks
    block_reduction(g_out, g_in, s_data, grid.size(), size, grid, block);

    grid.sync();

    // do reduction with single block
    if (block.group_index().x == 0) {
        block_reduction(g_out, g_out, s_data, block.size(), gridDim.x, grid, block);
    }
}

int reduction_grid_sync(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{   
    int num_blocks_per_sm;
    cudaDeviceProp deviceProp;

    // Calculate the device occupancy to know how many blocks can be run concurrently
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_kernel, n_threads, n_threads*sizeof(float));
    int num_sms = deviceProp.multiProcessorCount;
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    void *params[3];
    params[0] = (void*)&g_outPtr;
    params[1] = (void*)&g_inPtr;
    params[2] = (void*)&size;
    cudaLaunchCooperativeKernel((void*)reduction_kernel, n_blocks, n_threads, params, n_threads * sizeof(float), NULL);

    return n_blocks;
}
