#include <stdio.h>
#include <assert.h>

__global__ void warp_sync_test_kernel(int size)
{
    int x;
    int tid = threadIdx.x;

    // __syncthreads_and()
    x = __syncthreads_and(tid % 2); // some tids can be devided by 2 or not.
    assert(!x);

    x = __syncthreads_and(1); // all threads pass this barrior with true
    assert(x);

    // __synctherads__or()
    x = __syncthreads_or(tid % 2); // some tids can be devided by 2 or not.
    assert(x);

    x = __syncthreads_or(0); // all threads pass this barrior with false
    assert(!x);

    // __syncthread__count()
    x = __syncthreads_count(tid % 2);
    assert(x == size/2);
}

int main(int argc, char *argv[]) 
{
    cudaDeviceProp prop;
    cudaError_t e;
    int size = 64;

    // check minimum CUDA capability for __syncthreads_* instruction
    cudaGetDeviceProperties(&prop, 0);
    printf("CUDA Compute Capability: %d.%d\n", prop.major, prop.minor);
    if (prop.major < 2)
        printf("This devices does not support __syncthreads() instruction.\n");

    warp_sync_test_kernel<<<1, size>>>(size);
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    if (!e)
        printf("SUCCESS\n");
    else
        printf("cuda error: %s\n", cudaGetErrorString(e));
}