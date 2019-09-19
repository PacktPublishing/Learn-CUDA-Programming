#include <iostream>
#include <iomanip>
#include <curand_kernel.h>

#define BLOCK_DIM 512
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

int main(int argc, char *argv[])
{
    curandState_t *devStates;
    float *fp_random;
    unsigned int *np_random;

    int M = 3, N = 5;
    size_t size = M * N;

    // allocate random seed space
    cudaMalloc((void **)&devStates, sizeof(curandState) * size);

    // Initialize the states
    setup_kernel<<<(size + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>
                    (devStates);

    // random number generation
    std::cout << "Generated random numbers" << std::endl;
    cudaMallocManaged((void**)&np_random, sizeof(*np_random) * size);
    generate_kernel<<<(size + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>
                    (np_random, const_cast<curandState_t *>(devStates));
    cudaDeviceSynchronize();
    printMatrix(np_random, M, N);

    // reset the device state
    cudaMemset(devStates, 0, sizeof(curandState) * size);
    setup_kernel<<<(size + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>
                    (devStates);
    
    // uniform distributed random number generation
    std::cout << "Generated uniform random numbers" << std::endl;
    cudaMallocManaged((void**)&fp_random, sizeof(*fp_random) * size);
    generate_uniform_kernel<<<(size + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>
                    (fp_random, const_cast<curandState_t *>(devStates));
    cudaDeviceSynchronize();
    printMatrix(fp_random, M, N);

    // terminates curand device states
    cudaFree(devStates);
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