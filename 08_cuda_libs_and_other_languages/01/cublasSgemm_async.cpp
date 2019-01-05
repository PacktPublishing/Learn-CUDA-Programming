#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

float* getMatrix(const int n_col, const int n_row);
void printMatrix(const float *matrix, const int ldm, const int n);

int main()
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cudaStream_t stream;

    float *pf_A, *pf_B, *pf_C;
    float *df_A, *df_B, *df_C;
    int M, N, K;
    float alpha, beta;

    M = 4;
    N = 5;
    K = 6;
    alpha = 1.f;
    beta = 1.f;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS initialization failed" << std::endl;
        return EXIT_FAILURE;
    }

    srand(2019);
    pf_A = getMatrix(M, K);
    pf_B = getMatrix(K, N);
    pf_C = getMatrix(M, N);

    std::cout << "A:" << std::endl;
    printMatrix(pf_A, K, M);
    std::cout << "B:" << std::endl;
    printMatrix(pf_B, N, K);
    std::cout << "C:" << std::endl;
    printMatrix(pf_C, N, M);

    cudaMalloc((void **)&df_A, M * K * sizeof(float));
    cudaMalloc((void **)&df_B, K * N * sizeof(float));
    cudaMalloc((void **)&df_C, M * N * sizeof(float));

    cudaStat = cudaStreamCreate(&stream);

    cublasSetMatrixAsync(M, K, sizeof(*df_A), pf_A, M, df_A, M, stream);
    cublasSetMatrixAsync(K, N, sizeof(*df_B), pf_B, K, df_B, K, stream);
    cublasSetMatrixAsync(M, N, sizeof(*df_C), pf_B, M, df_C, M, stream);

    cublasSetStream(handle, stream);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        M, N, K, 
        &alpha, df_A, M,
        df_B, K, &beta, df_C, M);

    cublasGetMatrixAsync(M, N, sizeof(*df_C), df_C, M, pf_C, M, stream);
    cudaStreamSynchronize(stream);

    std::cout << "C out:" << std::endl;
    printMatrix(pf_C, N, M);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    cudaFree(df_A);
    cudaFree(df_B);
    cudaFree(df_C);

    free(pf_A);
    free(pf_B);
    free(pf_C);

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