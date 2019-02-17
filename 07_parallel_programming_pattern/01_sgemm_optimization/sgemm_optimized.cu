#include <stdio.h>
#include <helper_timer.h>
#include <cuda_profiler_api.h>

#define BLOCK_DIM 16

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on GPU
//! C = alpha * A * B + beta * C
//! @param A          matrix A as provided to device (M x K)
//! @param B          matrix B as provided to device (K x N)
//! @param C          matrix C as provided to device (M x N)
//! @param N          height of matrix A and matrix C
//! @param M          width of matrix B and matrix C
//! @param K          width of matrix A and height of matrix C
//! @param alpha      scala value for matrix multiplication
//! @param beta       scala value for matrix summation with C
////////////////////////////////////////////////////////////////////////////////

__global__ void sgemm_kernel(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta)
{
    int bid_x = blockIdx.x * blockDim.x;
    int bid_y = blockIdx.y * blockDim.y;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    __shared__ float s_tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float s_tile_B[BLOCK_DIM][BLOCK_DIM];

    float element_c = 0.f;

    // forward tile with tile size in matrix A
    for (int k = 0; k < K; k += BLOCK_DIM)
    {
        s_tile_A[tid_y][tid_x] = A[ (bid_y + tid_y) * K + k + tid_x ]; // Get sub-matrix from A
        s_tile_B[tid_y][tid_x] = B[ k * N + bid_x + tid_x ]; // Get sub-matrix from B

        __syncthreads();

        // compute gemm operation with tiles
        for (int e = 0; e < BLOCK_DIM; e++)
            element_c += s_tile_A[tid_y][e] * s_tile_B[e][tid_x];
    }

    C[(bid_y + tid_y) * N + (bid_x + tid_x)] = \
        alpha * element_c + beta * C[(bid_y + tid_y) * N + (bid_x + tid_x)];
}

void sgemm_gold(const float *A, const float *B, float *C, int M, int N, int K, float alpha, float beta)
{
    float element_c = 0.f;
    for (int row = 0; row < M; row++)
    {
        for (int col = 0; col < N; col++)
        {
            for (int e = 0; e < K; e++)
	    {
                element_c += A[row * K + e] * B[e * N + col];
	    }
            C[row * N + col] = alpha * element_c + beta * C[row * N + col];
        }
    }
}

void random_init(float *data, int length)
{
    for (int i = 0; i < length; i++)
    {
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

bool value_test(float *host, float *gpu, int length)
{
    float epsilon = 0.000001;
    bool result = true;
    for (int i = 0; i < length; i++)
        if (abs(host[i] - gpu[i]) >= epsilon)
            result = false;
    return result;
}

int main()
{
    float *A, *B, *C_host, *C_gpu;
    float *d_A, *d_B, *d_C;
    int N, M, K;
    float alpha = 2.f;
    float beta = 1.f;
    int n_iter = 5;
    N = M = K = 2048;

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    // allocation of linear memory space
    A = (float *)malloc(N * K * sizeof(float));
    B = (float *)malloc(K * M * sizeof(float));
    C_host = (float *)malloc(M * N * sizeof(float));
    C_gpu = (float *)malloc(M * N * sizeof(float));

    // allocation of gpu linear memory space
    cudaMalloc((void **)&d_A, N * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * M * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    // initialize randomized values for memory space
    random_init(A, N * K);
    random_init(B, K * M);

    // copy initial value for gpu memory
    cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, A, K * M * sizeof(float), cudaMemcpyHostToDevice);

    // Start to log timer
    sdkStartTimer(&timer);

    // do operation
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);
    cudaProfilerStart();
    for (int i = 0; i < n_iter; i++) {
        sgemm_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    // measuring the elapsed time
    cudaDeviceSynchronize();
    sdkStopTimer(&timer); // this profiler should be behined of device synchronization

    // copy data from the gpu
    cudaMemcpy(C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // profiler will stop its focus
    cudaProfilerStop();

    // compare the result
    sgemm_gold(A, B, C_host, M, N, K, alpha, beta);
    if (value_test(C_host, C_gpu, M * N))
        printf("SUCCESS!!\n");
    else
        printf("Error\n");

    // terminates allocated gpu memory space
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // terminates allocated memory space
    free(A);
    free(B);
    free(C_host);
    free(C_gpu);

    return 0;
}