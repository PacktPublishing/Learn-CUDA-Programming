#include <cstdio>
#include <cstdlib>

using namespace std;

#define BUF_SIZE (1 << 10)
#define BLOCKDIM 256

__global__ void child_kernel(int *data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    data[idx] += 1;
}

__global__ void parent_kernel(int *data, int n)
{
    if (threadIdx.x == 0) {
        child_kernel<<< BUF_SIZE/BLOCKDIM, BLOCKDIM >>>(&data[blockIdx.x * BUF_SIZE / BLOCKDIM]);
        // synchronization for child kernel output
        cudaDeviceSynchronize();
    }
    // synchronization for other parent's kernel output
    __syncthreads();
}

int main()
{
    int *data;

    cudaMallocManaged((void**)&data, BUF_SIZE * sizeof(int));

    parent_kernel<<<2, 1>>>(data, BUF_SIZE / 2);

    cudaDeviceSynchronize();
    
    // Count elements value
    int counter = 0;
    for (int i = 0; i < BUF_SIZE; i++) {
        counter += data[i];
    }

    if (BUF_SIZE == counter)
        printf("Correct!!\n");
    else
        printf("Error!! Obtained %d. It should be %d\n", counter, BUF_SIZE);

    cudaFree(data);

    return 0;
}

