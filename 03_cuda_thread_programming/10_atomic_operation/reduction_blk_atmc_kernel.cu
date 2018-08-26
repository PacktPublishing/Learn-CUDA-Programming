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

/**
    Two warp level primitives are used here for this example
    https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
    https://devblogs.nvidia.com/using-cuda-warp-level-primitives/

    Disadvantage in this approaches is floating point reduction will not be exact from run to run.
    https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
 */

template <typename group_t>
__inline__ __device__ float warp_reduce_sum(group_t group, float val)
{
    #pragma unroll
    for (int offset = group.size() / 2; offset > 0; offset >>= 1)
        val += group.shfl_down(val, offset);
    return val;
}

__inline__ __device__ float block_reduce_sum(thread_block block, float val)
{
    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int wid = threadIdx.x / warpSize;
    thread_block_tile<32> tile32 = tiled_partition<32>(block);

    val = warp_reduce_sum(tile32, val); // Each warp performs partial reduction

    if (tile32.thread_rank() == 0)
        shared[wid] = val; // Write reduced value to shared memory

    __syncthreads(); // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[tile32.thread_rank()] : 0;

    if (wid == 0)
        val = warp_reduce_sum(tile32, val); //Final reduce within first warp

    return val;
}

// large vector reduction
__global__ void
reduction_blk_atmc_kernel(float *g_out, float *g_in, unsigned int size)
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

    if (block.thread_rank() == 0) {
        atomicAdd(&g_out[0], sum[0]);
    }
}

void atomic_reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{   
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_blk_atmc_kernel, n_threads, n_threads*sizeof(float));
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    reduction_blk_atmc_kernel<<<n_blocks, n_threads>>>(g_outPtr, g_inPtr, size);
}
