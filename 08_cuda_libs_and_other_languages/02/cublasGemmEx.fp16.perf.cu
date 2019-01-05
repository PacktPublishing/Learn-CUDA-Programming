#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper.cuh"

#define BLOCKDIM 512

template <typename T>
void printMatrix(const T *matrix, const int ldm, const int n);

int main()
{
    cublasStatus_t stat;
    cublasHandle_t cublas_handle;
    cudaEvent_t start, stop;

    CBuffer<half> A, B;
    CBuffer<float> C;
    int M, N, K;
    float alpha, beta;

    M = 8192;
    N = 8192;
    K = 8192;
    alpha = 1.f;
    beta = 1.f;

    stat = cublasCreate(&cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS initialization failed" << std::endl;
        return EXIT_FAILURE;
    }

    srand(2019);

    // initialize host buffers
    std::cout << "Create matrices.." << std::endl;
    A.init(K * M, true);
    B.init(N * K, true);
    C.init(N * M, true);

    A.cuda(true);
    B.cuda(true);
    C.cuda(true);

    // create CUDA events to obtain performance data
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // enables tensorcore operation when it is possible
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

    std::cout << "Starts cuBLAS operation.." << std::endl;

    cudaEventRecord(start);
    stat = cublasGemmEx(cublas_handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        M, N, K,
                        &alpha,
                        A.d_ptr_, CUDA_R_16F, M,
                        B.d_ptr_, CUDA_R_16F, K,
                        &beta,
                        C.d_ptr_, CUDA_R_32F, M,
                        CUDA_R_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS operation failed [" << stat << "]" << std::endl;
        return EXIT_FAILURE;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // print out elapsed time
    float cudaElapsedTime;
    cudaEventElapsedTime(&cudaElapsedTime, start, stop);
    std::cout << std::setw(4) << cudaElapsedTime << " ms" << std::endl;

    cublasDestroy(cublas_handle);

    return 0;
}

template <typename T>
void printMatrix(const T *matrix, const int ldm, const int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < ldm; i++)
        {
            if (sizeof(T) >= 2)
                std::cout << std::fixed << std::setw(8) << std::setprecision(4) << __half2float(matrix[IDX2C(i, j, ldm)]);
            else
                std::cout << std::fixed << std::setw(4) << static_cast<int16_t>(matrix[IDX2C(i, j, ldm)]);
        }
        std::cout << std::endl;
    }
}