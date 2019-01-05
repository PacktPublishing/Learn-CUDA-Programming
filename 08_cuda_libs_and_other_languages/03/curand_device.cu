#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand_kernel.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

template <typename T>
void printMatrix(const T *matrix, const int ldm, const int n);

__global__ void setup_kernel(curandState_t *state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread gets same seed, 
    // a different sequence number, no offset */
    curand_init(2019UL, idx, 0, &state[idx]);
}

__global__ void generate_kernel(unsigned int *generated_out, curandState_t *state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    generated_out[idx] = curand(&state[idx]);
}

__global__ void generate_uniform_kernel(float *generated_out, curandState_t *state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    generated_out[idx] = curand_uniform(&state[idx]);
}

#define BLOCK_DIM 512

/*
 * random integer number generation
 */
void cuRandGenerator(const curandState_t *devStates, unsigned int **npHostResult, const size_t length)
{
    unsigned int *npDevResult;

    (*npHostResult) = new unsigned int[length];
    cudaMalloc((void **)&npDevResult, length * sizeof(unsigned int));

    // random number generation
    generate_kernel<<<(length + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(npDevResult, const_cast<curandState_t *>(devStates));
    cudaMemcpy(*npHostResult, npDevResult, length * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // terminates memory
    cudaFree(npDevResult);
}

/*
 * Random uniform distributed floating number generation
 */
void cuRandUniformGenerator(const curandState_t *devStates, float **fpHostResult, const size_t length)
{
    float *fpDevResult;

    (*fpHostResult) = new float[length];
    cudaMalloc((void **)&fpDevResult, length * sizeof(float));

    // random number generation
    generate_uniform_kernel<<<(length + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(fpDevResult, const_cast<curandState_t *>(devStates));
    cudaMemcpy(*fpHostResult, fpDevResult, length * sizeof(float), cudaMemcpyDeviceToHost);

    // terminates device memory
    cudaFree(fpDevResult);
}

int main(int argc, char *argv[])
{
    curandState_t *devStates;
    unsigned int *npHostResult;
    float *fpHostResult;
    int opt = 0;

    int M = 4, N = 5;
    size_t length = M * N;

    // Select random number generation option
    if (argc == 2) 
        opt = atoi(argv[1]);

    // allcate space for prng states on device
    cudaMalloc((void **)&devStates, length * sizeof(curandState));

    /* Initialize the states */
    setup_kernel<<<(length + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(devStates);

    if (opt == 0)
    {
        std::cout << "Generated random numbers" << std::endl;
        cuRandGenerator(devStates, &npHostResult, length);
        printMatrix(npHostResult, M, N);
        delete [] npHostResult;
    }
    else
    {
        std::cout << "Generated uniform random numbers" << std::endl;
        cuRandUniformGenerator(devStates, &fpHostResult, length);
        printMatrix(fpHostResult, M, N);
        delete [] fpHostResult;
    }

    // terminates curand device states
    cudaFree(devStates);

    return 0;
}

template <typename T>
void printMatrix(const T *matrix, const int ldm, const int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < ldm; i++)
            std::cout << std::fixed << std::setw(12) << std::setprecision(4) << matrix[IDX2C(i, j, ldm)];
        std::cout << std::endl;
    }
}