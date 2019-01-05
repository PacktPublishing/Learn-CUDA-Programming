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

    CBuffer<float> A, B, C;
    int M, N, K;
    float alpha, beta;

    M = 4;
    N = 5;
    K = 6;
    alpha = 1.f;
    beta = 0.f;

    stat = cublasCreate(&cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS initialization failed" << std::endl;
        return EXIT_FAILURE;
    }

    srand(2019);

    // initialize host buffers
    A.init(K * M, true);
    B.init(N * K, true);
    C.init(N * M, true);

    std::cout << "A:" << std::endl;
    printMatrix(A.h_ptr_, K, M);
    std::cout << "B:" << std::endl;
    printMatrix(B.h_ptr_, N, K);
    std::cout << "C:" << std::endl;
    printMatrix(C.h_ptr_, N, M);

    A.cuda(true);
    B.cuda(true);
    C.cuda(true);

    stat = cublasGemmEx(cublas_handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        M, N, K,
                        &alpha,
                        A.d_ptr_, CUDA_R_32F, M,
                        B.d_ptr_, CUDA_R_32F, K,
                        &beta,
                        C.d_ptr_, CUDA_R_32F, M,
                        CUDA_R_32F,
                        CUBLAS_GEMM_DEFAULT);

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS operation failed [" << stat << "]" << std::endl;
        return EXIT_FAILURE;
    }

    C.copyToHost();

    std::cout << "C out:" << std::endl;
    printMatrix(C.h_ptr_, N, M);

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
                std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[IDX2C(i, j, ldm)];
            else
                std::cout << std::fixed << std::setw(4) << static_cast<int16_t>(matrix[IDX2C(i, j, ldm)]);
        }
        std::cout << std::endl;
    }
}