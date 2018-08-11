#include <stdio.h>
#include "reduction.h"

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

__inline__ __device__ float warp_reduce_sum(float val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

__inline__ __device__ float block_reduce_sum(float val)
{
    static __shared__ float shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val); // Each warp performs partial reduction

    if (lane == 0)
        shared[wid] = val; // Write reduced value to shared memory

    __syncthreads(); // Wait for all partial reductions

    //read from shared memory only if that warp existed
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
        val = warp_reduce_sum(val); //Final reduce within first warp
    }

    return val;
}

// cuda thread synchronization
__global__ void
reduction_kernel(float *g_out, float *g_in, unsigned int size)
{
    unsigned int idx_x = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

    float sum = 0.f;
    // reduce one more data
    sum += (idx_x < size) ? g_in[idx_x] : 0.f;
    sum += ((idx_x + blockDim.x) < size) ? g_in[idx_x + blockDim.x] : 0.f;

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0)
        g_out[blockIdx.x] = sum;
}

int reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{
    int block_size = 2 * n_threads;
    int n_blocks = (size + block_size - 1) / block_size;
    reduction_kernel<<<n_blocks, n_threads>>>(g_outPtr, g_inPtr, size);
    return n_blocks;
}
