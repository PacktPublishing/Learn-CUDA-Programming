#include <cstdio>

using namespace std;

__global__ void
foo_kernel(int step)
{
    printf("loop: %d\n", step);
}

int main()
{
    int n_loop = 5;

    // execute kernels with the default stream
    for (int i = 0; i < n_loop; i++)
        foo_kernel<<< 1, 1, 0, 0 >>>(i);

    cudaDeviceSynchronize();

    return 0;
}