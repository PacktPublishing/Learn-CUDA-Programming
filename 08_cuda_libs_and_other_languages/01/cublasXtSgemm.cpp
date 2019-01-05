#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasXt.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

float* getMatrix(const int n_col, const int n_row);
void printMatrix(const float *matrix, const int ldm, const int n);

int main()
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasXtHandle_t handle;
    //cublasHandle_t handle;

    float *pf_A, *pf_B, *pf_C;
    float *df_A, *df_B, *df_C;
    int M, N, K;
    float alpha, beta;

    // cublasXt operation parameter
    int num_of_devices = 2;
    int num_of_total_devices;
    int *devices;

    M = 640;
    N = 320;
    K = 480;
    alpha = 1.f;
    beta = 1.f;

    // create CUDA event to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    stat = cublasXtCreate(&handle);
    //stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS initialization failed" << std::endl;
        return EXIT_FAILURE;
    }

    srand(2019);
    pf_A = getMatrix(M, K);
    pf_B = getMatrix(K, N);
    pf_C = getMatrix(M, N);

    cudaMalloc((void **)&df_A, M * K * sizeof(float));
    cudaMalloc((void **)&df_B, K * N * sizeof(float));
    cudaMalloc((void **)&df_C, M * N * sizeof(float));

    cublasSetMatrix(M, K, sizeof(*df_A), pf_A, M, df_A, M);
    cublasSetMatrix(K, N, sizeof(*df_B), pf_B, K, df_B, K);
    cublasSetMatrix(M, N, sizeof(*df_C), pf_B, M, df_C, M);

    cudaGetDeviceCount(&num_of_total_devices);
    devices = (int *)calloc(num_of_devices, sizeof(int));
    for (int i = 0; i < num_of_devices; i++)
        devices[i] = i;

    // select devices for use in cublasxt math
    cublasXtDeviceSelect(handle, num_of_devices, devices);

    cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                  M, N, K,
                  &alpha, df_A, M,
                  df_B, K, &beta, df_C, M);

    cudaEventRecord(start);

    cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                  M, N, K,
                  &alpha, df_A, M,
                  df_B, K, &beta, df_C, M);

    // cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //     M, N, K, 
    //     &alpha, df_A, M, df_B, K, &beta, df_C, M);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cublasGetMatrix(M, N, sizeof(*df_C), df_C, M, pf_C, M);

    // report execution time
    float elapsedTime = 0.f;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    float gFlops = 2 * M * N * K * 1e-9f / elapsedTime * 1e+3f;
    std::cout << "Elapsed Time on " << num_of_devices << " GPUs: " << elapsedTime << " ms, " << gFlops << " GFlops." << std::endl;
    std::cout << 2 * M * N * K << std::endl;

    // destory memories
    cudaFree(df_A);
    cudaFree(df_B);
    cudaFree(df_C);

    free(pf_A);
    free(pf_B);
    free(pf_C);
    free(devices);

   cublasXtDestroy(handle);
    // cublasDestroy(handle);

    return 0;
}

float* getMatrix(const int ldm, const int n)
{
    float *pf_matrix = (float *)malloc(ldm * n * sizeof(float));

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < ldm; i++)
        {
            pf_matrix[IDX2C(i, j, ldm)] = (float)rand() / RAND_MAX;
        }
    }

    return pf_matrix;
}

void printMatrix(const float* matrix, const int ldm, const int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < ldm; i++)
        {
            std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[IDX2C(i, j, ldm)];
        }
        std::cout << std::endl;
    }
}