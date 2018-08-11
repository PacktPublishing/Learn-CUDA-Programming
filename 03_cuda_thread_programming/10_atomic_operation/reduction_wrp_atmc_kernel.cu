#include <stdio.h>
#include <cooperative_groups.h>
#include "reduction.h"

//namespace cg = cooperative_groups;
using namespace cooperative_groups;

#define FULL_MASK 0xffffffff

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

template <typename group_t>
__inline__ __device__ float warp_reduce_sum(group_t group, float val)
{
    #pragma unroll
    for (int offset = group.size() / 2; offset > 0; offset >>= 1)
        val += group.shfl_down(val, offset);
    return val;
}

// large vector reduction
__global__ void
reduction_wrp_atmc_kernel(float *g_out, float *g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    thread_block block = this_thread_block();
    thread_block_tile<32> tile32 = tiled_partition<32>(block);

    // cumulates input with grid-stride loop and save to share memory
    float sum = 0.f;
    for (int i = idx_x; i < size; i += blockDim.x * gridDim.x)
        sum += g_in[i];
    // warp synchronous reduction
    sum = warp_reduce_sum(tile32, sum);

    //if ((threadIdx.x & (warpSize - 1)) == 0)
    if (tile32.thread_rank() == 0)
        atomicAdd(&g_out[0], sum);;
}

void reduction_wrp_atmc(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{   
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_wrp_atmc_kernel, n_threads, n_threads*sizeof(float));
    int n_blocks = min(num_blocks_per_sm * num_sms, (size + n_threads - 1) / n_threads);

    reduction_wrp_atmc_kernel<<<n_blocks, n_threads>>>(g_outPtr, g_inPtr, size);
}
