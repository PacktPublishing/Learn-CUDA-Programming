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

template <typename group_t>
__inline__ __device__ float warp_reduce_sum(group_t group, float val)
{
    #pragma unroll 5
    for (int offset = group.size() / 2; offset > 0; offset >>= 1)
        val += group.shfl_down(val, offset);
    return val;
}

__inline__ __device__ float block_reduce_sum(thread_block block, float val)
{
    __shared__ float shared[32]; // Shared mem for 32 partial sums
    // int warp_idx = threadIdx.x / warpSize;
    int warp_idx = block.thread_index().x / warpSize;

    // partial reduciton at tile<32> size
    thread_block_tile<32> tile32 = tiled_partition<32>(block);
    val = warp_reduce_sum(tile32, val);

    // write reduced value to shared memory
    if (tile32.thread_rank() == 0)
        shared[warp_idx] = val; 

    block.sync(); // Wait for all partial reductions

    //read from shared memory only if that warp existed
    if (warp_idx == 0) {
        val = (block.thread_index().x < blockDim.x / warpSize) ? shared[tile32.thread_rank()] : 0;
        val = warp_reduce_sum(tile32, val); //Final reduce within first warp
    }

    return val;
}

// large vector reduction
__global__ void
reduction_kernel(float* g_out, float* g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    thread_block block = this_thread_block();

    // cumulates input with grid-stride loop and save to share memory
    float sum[NUM_LOAD] = { 0.f };
    for (int i = idx_x; i < size; i += blockDim.x * gridDim.x * NUM_LOAD)
    {
        for (int step = 0; step < NUM_LOAD; step++)
            sum[step] += (i + step * blockDim.x * gridDim.x < size) ? g_in[i + step * blockDim.x * gridDim.x] : 0.f;
    }
    for (int i = 1; i < NUM_LOAD; i++)
        sum[0] += sum[i];
        
    // warp synchronous reduction
    sum[0] = block_reduce_sum(block, sum[0]);

    if (block.thread_index().x == 0)
        g_out[block.group_index().x] = sum[0];
}

void reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{   
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_kernel, n_threads, n_threads*sizeof(float));
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    reduction_kernel<<< n_blocks, n_threads>>>(g_outPtr, g_inPtr, size);
    reduction_kernel<<< 1, n_threads >>>(g_outPtr, g_inPtr, n_blocks);
}
