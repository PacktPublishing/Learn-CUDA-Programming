#include <cstdio>
#include <cstdlib>

using namespace std;

#define BUFSIZE 256

__global__ void child_kernel(int *data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    data[idx] = 1;
}

__global__ void parent_kernel(int *data)
{
    if (threadIdx.x == 0)
        child_kernel<<< 4, BUFSIZE / 4 >>>(data);
    cudaDeviceSynchronize();
}

int main()
{
    int *data;

    cudaMallocManaged((void**)&data, BUFSIZE * sizeof(int));

    parent_kernel<<<1, 1>>>(data);
    
    // Count elements value
    int counter = 0;
    for (int i = 0; i < BUFSIZE; i++) {
        counter += data[i];
    }

    if (BUFSIZE == counter)
        printf("Correct!!\n");
    else
        printf("Error!! Obtained %d. It should be %d\n", counter, BUFSIZE);

    cudaFree(data);

    return 0;
}

