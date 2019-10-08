#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <curand.h>
#include "helper.cuh"

typedef half    Real;
typedef half2   Complex;

int main(int argc, char *argv[])
{
    long long sample_size = 1 << 20;      // 1,048,576
    const int batch_size = 1 << 9;  // 512

    cufftHandle plan_forward, plan_inverse;
    Real    *p_sample;
    Complex *d_freq;

    float forward_time_ms, inverse_time_ms;

    // create cuda event to measure the performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // create host buffer as input data
    cudaMallocManaged((void**)&p_sample, sizeof(Real) * sample_size * batch_size);

    // create curand generator & set random seed
    curandGenerator_t curand_gen;
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 2019UL);
    op::curand(curand_gen, p_sample, sample_size * batch_size);

    // create signal and filter memory
    cudaMalloc((void**)&d_freq,   sample_size * sizeof(Complex) * batch_size);

    // 1D cufft setup
    int rank = 1;
    int stride_sample = 1, stride_freq = 1;
    long long int dist_sample = sample_size, dist_freq = sample_size / 2 + 1;
    long long embed_sample[] = {0};
    long long embed_freq[] = {0};
    size_t workSize = 0;
    cufftCreate(&plan_forward);
    cufftXtMakePlanMany(plan_forward, 
        rank, &sample_size, 
        embed_sample, stride_sample, dist_sample, CUDA_R_16F, 
        embed_freq, stride_freq, dist_freq, CUDA_C_16F, 
        batch_size, &workSize, CUDA_C_16F);
    cufftCreate(&plan_inverse);
    cufftXtMakePlanMany(plan_inverse,
    	rank, &sample_size,
        embed_freq, stride_freq, dist_freq, CUDA_C_16F,
        embed_sample, stride_sample, dist_sample, CUDA_R_16F,
        batch_size, &workSize, CUDA_R_16F);

    // executes FFT processes
    cudaEventRecord(start);
    cufftXtExec(plan_forward, p_sample, d_freq, CUFFT_FORWARD);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&forward_time_ms, start, stop);

    // executes FFT processes (inverse transformation)
    cudaEventRecord(start);
    cufftXtExec(plan_inverse, d_freq, p_sample, CUFFT_INVERSE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&inverse_time_ms, start, stop);

    // print elapsed time
    std::cout << "FFT operation time for " << sample_size << " elements with " << batch_size << " batch.." << std::endl;
    std::cout << "Forward (ms): " << forward_time_ms << std::endl;
    std::cout << "Inverse (ms): " << inverse_time_ms << std::endl;

    // terminates used resources
    curandDestroyGenerator(curand_gen);

    // deletes CUFFT plan_forward
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);

    // terminates memories
    cudaFree(p_sample);
    
    // delete cuda events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
