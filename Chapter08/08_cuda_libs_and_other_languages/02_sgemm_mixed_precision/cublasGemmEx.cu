#include <iostream>
#include <iomanip>
#include <cublas_v2.h>
#include "helper_cuda.h"
#include "helper.cuh"

using namespace helper;

int main()
{
    int M, N, K;
    M = 4;
    N = 5;
    K = 6;

    srand(2019);
    // initialize host buffers
    CBuffer<half> A, B;
    CBuffer<float> C;
    float alpha, beta;

    A.init(K * M, true);
    B.init(N * K, true);
    C.init(N * M, true);

    bool tensor_core = false;

    // create cuda event handles
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // create cublas handle
    cublasHandle_t cublas_handle;
    checkCudaErrors(
        cublasCreate(&cublas_handle));

    int print_threshold = 12;
    if (M < print_threshold && N < print_threshold && K < print_threshold) {
        std::cout << "A:" << std::endl;
        printMatrix(A.h_ptr_, K, M);
        std::cout << "B:" << std::endl;
        printMatrix(B.h_ptr_, N, K);
        std::cout << "C:" << std::endl;
        printMatrix(C.h_ptr_, N, M);
    }

    alpha = 1.f;
    beta = 0.f;

    // determin data type information for GemmEx()
    cudaDataType TYPE_A, TYPE_B, TYPE_C;
    if (typeid(*A.h_ptr_) == typeid(float)) {
        TYPE_A = TYPE_B = CUDA_R_32F;
    }
    else if (typeid(*A.h_ptr_) == typeid(half)) {
        TYPE_A = TYPE_B = CUDA_R_16F;
    }
    else if (typeid(*A.h_ptr_) == typeid(int8_t)) {
        TYPE_A = TYPE_B = CUDA_R_8I;
    }
    else {
        printf("Not supported precision\n");
        return -1;
    }

    if (typeid(*C.h_ptr_) == typeid(float)) {
        TYPE_C = CUDA_R_32F;
    }
    else if (typeid(*C.h_ptr_) == typeid(int)) {
        TYPE_C = CUDA_R_32I;
    }
    else {
        printf("Not supported precision\n");
        return -1;
    }

    // allocate GPU memory and copy the data
    A.cuda(true);
    B.cuda(true);
    C.cuda(true);

    // enables tensorcore operation when it is possible
    // checkCudaErrors(
    //     cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    cudaEventRecord(start);
    checkCudaErrors(
        cublasGemmEx(cublas_handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        M, N, K,
                        &alpha,
                        A.d_ptr_, TYPE_A, M,
                        B.d_ptr_, TYPE_B, K,
                        &beta,
                        C.d_ptr_, TYPE_C, M,
                        TYPE_C,
                        (tensor_core) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
    cudaEventRecord(stop);

    C.copyToHost();

    if (M < print_threshold && N < print_threshold && K < print_threshold) {
        std::cout << "C out:" << std::endl;
        printMatrix(C.h_ptr_, N, M);
    }

    // print out elapsed time
    float cudaElapsedTime;
    cudaEventElapsedTime(&cudaElapsedTime, start, stop);
    std::cout << std::setw(4) << cudaElapsedTime << " ms" << std::endl;

    checkCudaErrors(
        cublasDestroy(cublas_handle));

    return 0;
}
