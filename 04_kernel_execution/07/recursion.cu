#include <cstdio>
#include <cstdlib>

using namespace std;

__global__ void recursive_kernel(int *data, int size, int depth)
{
    int x_0 = blockIdx.x * size;

    int idx = x_0 + threadIdx.x;
    if (threadIdx.x < size)
        data[idx] += depth;

    if (depth > 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            int dimBlock = 256;
            int dimGrid = size / dimBlock;
            recursive_kernel<<< dimGrid, dimBlock>>>(&data[x_0], size / dimGrid, depth - 1);
            cudaDeviceSynchronize();
        }
        __syncthreads();
    }
}

int sum_depth(int depth) {
    if (depth == 1)
        return 1;
    return sum_depth(depth - 1) + depth;
}

int main()
{
    int *data;
    int size = 1 << 20;
    int max_depth = 3;

    cudaMallocManaged((void**)&data, size * sizeof(int));

    int dimBlock = 256;
    int dimGrid = size / dimBlock;
    recursive_kernel<<< dimGrid, dimBlock>>>(data, size / dimGrid, max_depth);

    cudaDeviceSynchronize();
    
    // Count elements value
    int counter = 0;
    for (int i = 0; i < size; i++) {
        counter += data[i];
    }

    printf("sum_depth: %d\n", sum_depth(max_depth));
    if (counter == size * sum_depth(max_depth))
        printf("Correct!!\n");
    else
        printf("Error!! Obtained %d. It should be %d\n", counter, size * sum_depth(max_depth));

    cudaFree(data);

    return 0;
}

