#include <stdio.h>
#include <cuda_profiler_api.h>
#include "nvToolsExt.h"

#define BLOCK_DIM 16

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on GPU
//! C = alpha * A * B + beta * C
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param C          matrix C as provided to device
//! @param N          height of matrix A and matrix C
//! @param M          width of matrix B and matrix C
//! @param K          width of matrix A and height of matrix C
//! @param alpha      scala value for matrix multiplication
//! @param beta       scala value for matrix summation with C
////////////////////////////////////////////////////////////////////////////////
__global__ void sgemm_kernel_A(const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    float sum = 0.f;
    for (int i = 0; i < K; ++i) 
        sum += A[row * K + i] * B[i * K + col];

    C[row * M + col] = alpha * sum + beta * C[row * M + col];
}

void sgemm_gpu_A(const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid(M / dimBlock.x, N / dimBlock.y);
    sgemm_kernel_A<<<dimGrid, dimBlock>>>(A, B, C, N, M, K, alpha, beta);
}

__global__ void sgemm_kernel_B(const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.f;
    for (int i = 0; i < K; ++i)
        sum += A[row * K + i] * B[i * K + col];

    C[row * M + col] = alpha * sum + beta * C[row * M + col];
}

void sgemm_gpu_B(const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid(M / dimBlock.x, N / dimBlock.y);
    sgemm_kernel_B<<<dimGrid, dimBlock>>>(A, B, C, N, M, K, alpha, beta);
}

void random_init(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

int main()
{
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    int N, M, K;
    float alpha = 2.f;
    float beta = 1.f;
    int n_iter = 5;
    N = M = K = 2048;

    // allocation of linear memory space
    A = (float *)malloc(N * K * sizeof(float));
    B = (float *)malloc(K * M * sizeof(float));
    C = (float *)malloc(N * M * sizeof(float));

    // allocation of gpu linear memory space
    nvtxRangePushA("Data Initialization");
    cudaMalloc((void **)&d_A, N * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * M * sizeof(float));
    cudaMalloc((void **)&d_C, N * M * sizeof(float));
    nvtxRangePop();

    // initialize randomized values for memory space
    nvtxRangePushA("Data Initialization");
    random_init(A, N * K);
    random_init(B, K * M);
    random_init(C, N * M);
    nvtxRangePop();

    cudaProfilerStart();
    
    // copy initial value for gpu memory
    nvtxRangePushA("Data Transfer");
    cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, A, K * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    nvtxRangePop();

    nvtxRangePushA("Kernel Execution");
    // do operation
    nvtxRangePushA("Kernel A");
    for (int i = 0; i < n_iter; i++)
        sgemm_gpu_A(d_A, d_B, d_C, N, M, K, alpha, beta);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePushA("Kernel B");
    for (int i = 0; i < n_iter; i++)
        sgemm_gpu_B(d_A, d_B, d_C, N, M, K, alpha, beta);
    cudaDeviceSynchronize();

    nvtxRangePop();
    nvtxRangePop();

    cudaProfilerStop();

    // terminates allocated gpu memory space
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // terminates allocated memory space
    free(A);
    free(B);
    free(C);

    return 0;
}