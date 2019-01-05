
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

template <typename T>
void printMatrix(const T *matrix, const int ldm, const int n);

/*
 * random integer number generation
 */
void cuRandGenerator(const curandGenerator_t gen, unsigned int **npHostResult, const size_t length)
{
    unsigned int *npDevResult;

    (*npHostResult) = new unsigned int[length];
    cudaMalloc((void **)&npDevResult, length * sizeof(unsigned int));

    // random number generation
    curandGenerate(gen, npDevResult, length);
    cudaMemcpy(*npHostResult, npDevResult, length * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // terminates memory
    cudaFree(npDevResult);
}

/*
 * Random uniform distributed floating number generation
 */
void cuRandUniformGenerator(const curandGenerator_t gen, float **fpHostResult, const size_t length)
{
    float *fpDevResult;
    (*fpHostResult) = new float[length];
    cudaMalloc((void **)&fpDevResult, length * sizeof(float));

    // random number generation
    curandGenerateUniform(gen, fpDevResult, length);
    cudaMemcpy(*fpHostResult, fpDevResult, length * sizeof(float), cudaMemcpyDeviceToHost);

    // terminates memory
    cudaFree(fpDevResult);
}

int main(int argc, char *argv[])
{
    curandGenerator_t curand_gen;
    unsigned int *npDevResult, *npHostResult;
    float *fpDevResult, *fpHostResult;
    int opt = 0;

    int M = 4, N = 5;
    size_t length = M * N;

    if (argc == 2)
        opt = atoi(argv[1]);

    // create curand generator & set random seed
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 2019UL);

    if (opt == 0) {
        std::cout << "Generated random numbers" << std::endl;
        cuRandGenerator(curand_gen, &npHostResult, length);
        printMatrix(npHostResult, M, N);
        delete [] npHostResult;
    }
    else {
        std::cout << "Generated uniform random numbers" << std::endl;
        cuRandUniformGenerator(curand_gen, &fpHostResult, length);
        printMatrix(fpHostResult, M, N);
        delete [] fpHostResult;
    }

    // terminates used resources
    curandDestroyGenerator(curand_gen);
    cudaFree(npDevResult);
    cudaFree(fpDevResult);

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