
// CUBLAS
// CUDNN
// WMMA

/**
 * Introduction to Tensor Cores
 * You can write a program to use Tensor Cores in CUDA C++.
 * Set of functions and types in the nvcuda::wmma namespaces
 * Tensor Cores are used concurrently by a full warp, This allows the warp to peform a 16x16x16 MMA at very high throughput
 * 
 * This simple receipt will show how you can use WMMA (Warp Matrix Multiply Accumlate) API to perform a matrix multiplication.
 * Related CUDA sample is named cudatensorCoreGemm. If you need to get highest performance then you should use cuBLAS.
 */

// https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/
// https://github.com/parallel-forall/code-samples/tree/master/posts/tensor-cores

#include <mma.h>
#include <cstdio>
#include <helper_timer.h>
#include "util.cuh"
using namespace nvcuda;

// The only dimenssions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_example(half *a, half *b, float *c,
                             int M, int N, int K,
                             float alpha, float beta)
{
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;
    
    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // fill the accumulator fragment with zeros
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over the K-dimension
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in current value of c, scale by beta, and add to result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
        
        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}

int main() 
{
    CBuffer<half> A, B;
    CBuffer<float> C;
    int M, N, K;
    M = N = K = 1 << 12;
    float alpha, beta;
    alpha = beta = 1.5f;
    int n_iteration = 100;

    // initialize input buffer
    A.init(true);
    B.init(true);
    C.init(true);

    // initialize gpu memories
    A.cuda(true);
    B.cuda(true);
    C.cuda(true);

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    dim3 dimBlock(16, 16);
    dim3 dimGrid(M/16, N/16);
    // wmma_example<<<dimGrid, dimBlock>>>(A.d_ptr_, B.d_ptr_, C.d_ptr_,
    //                          M, N, K, alpha, beta);

    

    for (int i = 0; i < n_iteration; i++)
        wmma_example<<<dimGrid, dimBlock>>>(A.d_ptr_, B.d_ptr_, C.d_ptr_,
                             M, N, K, alpha, beta);

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    float elapsedTimeMs = sdkGetTimerValue(&timer) / (float)n_iteration;
    float throughput = 3 * 2.f * M * N * K / elapsedTimeMs;
    printf("error:%d\n", cudaGetLastError());
    printf("FMA, Throughput = %.3f GFlops, Operation Time= %.3f msec\n", throughput * 1e-6, elapsedTimeMs);

    // cleanup
    sdkDeleteTimer(&timer);
    
    return 0;
}