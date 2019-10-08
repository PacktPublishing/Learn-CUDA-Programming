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
    cublasXtHandle_t handle;

    float *A, *B, *C;
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

    // create handle
    cublasXtCreate(&handle);

    srand(2019);
    A = getMatrix(M, K);
    B = getMatrix(K, N);
    C = getMatrix(M, N);

    cudaGetDeviceCount(&num_of_total_devices);
    devices = (int *)calloc(num_of_devices, sizeof(int));
    for (int i = 0; i < num_of_devices; i++)
        devices[i] = i;

    // select devices for use in cublasxt math
    cublasXtDeviceSelect(handle, num_of_devices, devices);

    // warm-up
    cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                  M, N, K,
                  &alpha, A, M,
                  B, K, &beta, C, M);

    cudaEventRecord(start);

    cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                  M, N, K,
                  &alpha, A, M,
                  B, K, &beta, C, M);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // report execution time
    float elapsedTime = 0.f;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    float gFlops = 2 * M * N * K * 1e-9f / elapsedTime * 1e+3f;
    std::cout << "Elapsed Time on " << num_of_devices << " GPUs: " << elapsedTime << " ms, " << gFlops << " GFlops." << std::endl;
    std::cout << 2 * M * N * K << std::endl;

    // destory memories
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    free(devices);

    cublasXtDestroy(handle);

    return 0;
}

float* getMatrix(const int ldm, const int n)
{
    float *pf_matrix = nullptr;
    cudaMallocManaged((void**)&pf_matrix, sizeof(float) * ldm * n);

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