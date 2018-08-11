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

/**
    Two warp level primitives are used here for this example
    https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
    https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
 */

template <unsigned size>
__inline__ __device__ float tile_reduce_sum(thread_block_tile<size> tile, float val)
{
    for (int offset = tile.size() / 2; offset > 0; offset >>= 1)
         val += tile.shfl_down(val, offset);
    return val;
}

__inline__ __device__ float block_reduce_sum(thread_block block, float val)
{
    __shared__ float shared[32]; // Shared mem for 32 partial sums
    int warp_idx = threadIdx.x / warpSize;

    // partial reduciton at tile<32> size
    thread_block_tile<32> tile32 = tiled_partition<32>(block);
    val = tile_reduce_sum(tile32, val);

    // write reduced value to shared memory
    if (tile32.thread_rank() == 0)
        shared[warp_idx] = val; 

    block.sync(); // Wait for all partial reductions

    //read from shared memory only if that warp existed
    if (warp_idx == 0) {
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[tile32.thread_rank()] : 0;
        val = tile_reduce_sum(tile32, val); //Final reduce within first warp
    }

    return val;
}

// cuda thread synchronization
__global__ void
reduction_kernel(float *g_out, float *g_in, unsigned int size)
{
    thread_block block = this_thread_block();
    unsigned int idx_x = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    
    float sum = 0.f;
    // reduce one more data
    sum += (idx_x < size) ? g_in[idx_x] : 0.f;
    sum += ((idx_x + block.size()) < size) ? g_in[idx_x + block.size()] : 0.f;

    sum = block_reduce_sum(block, sum);

    if (block.thread_index().x == 0)
        g_out[block.group_index().x] = sum;
}

int reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{
    int block_size = 2 * n_threads;
    int n_blocks = (size + block_size - 1) / block_size;
    reduction_kernel<<<n_blocks, n_threads>>>(g_outPtr, g_inPtr, size);
    return n_blocks;
}
