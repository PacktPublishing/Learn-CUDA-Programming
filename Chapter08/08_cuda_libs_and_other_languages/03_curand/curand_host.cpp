
#include <iostream>
#include <iomanip>
#include <curand.h>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

template <typename T>
void printMatrix(const T *matrix, const int ldm, const int n);

int main(int argc, char *argv[])
{
    int M = 3, N = 5;

    // initialize memory space for random numbers
    size_t size = M * N;
    unsigned int *np_random;
    float *fp_random;
    cudaMallocManaged((void**)&np_random, sizeof(*np_random) * size);
    cudaMallocManaged((void**)&fp_random, sizeof(*fp_random) * size);

    // create curand generator & set random seed
    curandGenerator_t curand_gen;
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 2019UL);

    // random number generation
    std::cout << "Generated random numbers" << std::endl;
    curandGenerate(curand_gen, np_random, size);
    cudaDeviceSynchronize();
    printMatrix(np_random, M, N);

    // reset the curand_gen state
    curandDestroyGenerator(curand_gen);
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 2019UL);

    // uniform distributed random number generation
    std::cout << "Generated uniform random numbers" << std::endl;
    curandGenerateUniform(curand_gen, fp_random, size);
    cudaDeviceSynchronize();
    printMatrix(fp_random, M, N);

    // terminates used resources
    curandDestroyGenerator(curand_gen);
    cudaFree(np_random);
    cudaFree(fp_random);

    return 0;
}

template <typename T>
void printMatrix(const T *matrix, const int ldm, const int n)
{
    for (int j = 0; j < ldm; j++)
    {
        for (int i = 0; i < n; i++)
            std::cout << std::fixed << std::setw(12) << std::setprecision(4) << matrix[IDX2C(i, j, ldm)];
        std::cout << std::endl;
    }
}