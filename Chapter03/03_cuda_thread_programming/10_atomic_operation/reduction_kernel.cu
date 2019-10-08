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

__global__ void
atomic_reduction_kernel(float *data_out, float *data_in, int size)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    atomicAdd(&data_out[0], data_in[idx_x]);
}

void atomic_reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads)
{
    int n_blocks = (size + n_threads - 1) / n_threads;
    atomic_reduction_kernel<<<n_blocks, n_threads>>>(g_outPtr, g_inPtr, size);
}
