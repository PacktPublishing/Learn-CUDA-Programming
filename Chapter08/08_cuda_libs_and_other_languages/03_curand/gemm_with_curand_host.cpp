#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include "fp16.cuh"

template <typename T>
typename std::enable_if<(std::is_same<T, float>::value), float>::type
*curand(curandGenerator_t generator, size_t length)
{
    T *buffer = nullptr;
    cudaMalloc((void **)&buffer, length * sizeof(float));
    curandGenerateUniform(generator, buffer, length);
    return buffer;
}

template <typename T>
typename std::enable_if<std::is_same<T, half>::value, half>::type
*curand(curandGenerator_t generator, size_t length)
{
    T *buffer = nullptr;
    float *buffer_fp32;

    cudaMalloc((void **)&buffer_fp32, length * sizeof(float));
    curandGenerateUniform(generator, buffer_fp32, length);

    cudaMalloc((void **)&buffer, length * sizeof(T));
    fp16::float2half(buffer, buffer_fp32, length);
    cudaFree(buffer_fp32);

    return buffer;
}

int main()
{
    cublasStatus_t stat;
    cublasHandle_t cublas_handle;
    curandGenerator_t curand_gen;
    cudaDeviceProp dev_prop;
    bool tensor_core = true;

    cudaEvent_t start, stop;

    int M, N, K;
    float alpha, beta;

    void *d_A, *d_B, *d_C;
    cudaDataType AType, BType, CType, computeType;
    M = 8192;
    N = 8192;
    K = 8192;
    alpha = 1.f;
    beta = 1.f;

    // operation option
    std::string precision = "fp16";

    // create curand generator
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 2019UL);

    // create cublas handler
    stat = cublasCreate(&cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS initialization failed" << std::endl;
        return EXIT_FAILURE;
    }

    // create cuda event to measure elapse time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // idendiify device architecture
    cudaGetDeviceProperties(&dev_prop, 0);
    cublasGemmAlgo_t gemm_algo = (tensor_core) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;

    if (precision == "fp32")
    {
        auto *a = curand<float>(curand_gen, M * K);
        auto *b = curand<float>(curand_gen, K * N);
        auto *c = curand<float>(curand_gen, M * N);
        AType = BType = CType = CUDA_R_32F;
        computeType = CUDA_R_32F;
        d_A = a, d_B = b, d_C = c;
    }
    else if (precision == "fp16")
    {
        auto *a = curand<half>(curand_gen, M * K);
        auto *b = curand<half>(curand_gen, K * N);
        auto *c = curand<float>(curand_gen, M * N);
        AType = BType = CUDA_R_16F, CType = CUDA_R_32F;
        computeType = CUDA_R_32F;
        d_A = a, d_B = b, d_C = c;
    }
    else
    {
        std::cout << "Sorry! it's undefined precision.." << std::endl;
        curandDestroyGenerator(curand_gen);
        cublasDestroy(cublas_handle);
        exit(EXIT_FAILURE);
    }

    // enable tensorcore operation when it is possible
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

    // record start time
    cudaEventRecord(start);

    // Gemm operation
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 M, N, K,
                 &alpha,
                 d_A, AType, M,
                 d_B, BType, K,
                 &beta,
                 d_C, CType, M,
                 computeType, gemm_algo);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS operation failed [" << stat << "]" << std::endl;
        return EXIT_FAILURE;
    }

    // record operation finish time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // print out elapsed time
    float cudaElapsedTime;
    cudaEventElapsedTime(&cudaElapsedTime, start, stop);
    std::cout << std::setw(4) << "Elpased Time: " << cudaElapsedTime << " ms" << std::endl;

    // terminates used resources
    curandDestroyGenerator(curand_gen);
    cublasDestroy(cublas_handle);

    return 0;
}
