#include <cstdio>
#include <cstdlib>

using namespace std;

__global__ void recursive_kernel(int *data, int block_size, int depth)
{
    if (depth > 24) {
        printf("CUDA does not support more than 24 depth recursion.\n");
        return;
    }

    int x_0 = blockIdx.x * block_size;

    int idx = x_0 + threadIdx.x;
    if (threadIdx.x < block_size)
        data[idx] += depth;

    if (depth > 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            int dimBlock = max(block_size/2, 32);
            int dimGrid = block_size / dimBlock;

            // prints the calling kernel information
            printf("depth: [%2d], offset: %4d, block_idx: %2d, block_size: %3d\n", 
                depth, x_0, blockIdx.x, block_size);

            recursive_kernel<<< dimGrid, dimBlock>>>(&data[x_0], dimBlock, depth - 1);
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
    int size = 1 << 9;
    int max_depth = 3;

    cudaMallocManaged((void**)&data, size * sizeof(int));

    int dimBlock = 512;
    int dimGrid = size / dimBlock;
    recursive_kernel<<< dimGrid, dimBlock>>>(data, dimBlock, max_depth);

    cudaDeviceSynchronize();
    
    // count elements value
    int counter = 0;
    int correct_sum = sum_depth(max_depth);
    for (int i = 0; i < size; i++) {
        counter += (data[i] == correct_sum) ? 1 : 0;
    }

    // result
    printf("sum_depth: %d\n", correct_sum);
    if (counter == size)
        printf("Correct!!\n");
    else
        printf("Error!! Obtained %d. It should be %d\n", counter, size);

    cudaFree(data);

    return 0;
}

