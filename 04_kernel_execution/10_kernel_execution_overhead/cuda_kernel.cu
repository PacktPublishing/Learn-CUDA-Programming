#include <cstdio>

using namespace std;

__global__ void
simple_saxpy_kernel(float *y, const float* x, const float alpha, const float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    y[idx] = alpha * x[idx] + beta;
}

__global__ void
iterative_saxpy_kernel(float *y, const float* x, const float alpha, const float beta, int n_loop)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < n_loop; i++)
        y[idx] = alpha * x[idx] + beta;
}

__global__ void
recursive_saxpy_kernel(float *y, const float* x, const float alpha, const float beta, int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (depth == 0)
        return;
    else 
        y[idx] = alpha * x[idx] + beta;
    
    if (threadIdx.x == 0) 
        recursive_saxpy_kernel<<< 1, blockDim.x >>>(y, x, alpha, beta, depth - 1);
}

int main()
{
    float *d_y, *d_x;
    int size = 1 << 10;
    int bufsize = size * sizeof(float);
    int n_loop = 24;
    float elapsed_time_A, elapsed_time_B, elapsed_time_C;
    float alpha = 0.1f, beta = 0.2f;

    cudaEvent_t start, stop;

    // initialize cuda event
    cudaEventCreateWithFlags(&start, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);

    cudaMalloc((void**)&d_y, bufsize);
    cudaMalloc((void**)&d_x, bufsize);

    int dimBlock = 256;
    int dimGrid = size / dimBlock;

    // Step 1. Loop outside the kernel
    cudaEventRecord(start);
    for (int i = 0; i < n_loop; i++) {
        simple_saxpy_kernel<<< dimGrid, dimBlock >>>(d_y, d_x, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_A, start, stop);

    // Step 2. Loop inside the kernel
    cudaEventRecord(start);
    iterative_saxpy_kernel<<< dimGrid, dimBlock >>>(d_y, d_x, alpha, beta, n_loop);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_B, start, stop);

    // Step 3. Loop with the recursion
    cudaEventRecord(start);
    recursive_saxpy_kernel<<< dimGrid, dimBlock >>>(d_y, d_x, alpha, beta, n_loop);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_C, start, stop);

    printf("Elapsed Time...\n");
    printf("simple loop: %.3f ms\n", elapsed_time_A);
    printf("inner loop : %.3f ms\n", elapsed_time_B);
    printf("recursion  : %.3f ms\n", elapsed_time_C);

    cudaFree(d_y);
    cudaFree(d_x);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}