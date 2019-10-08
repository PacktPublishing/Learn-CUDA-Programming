#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>

typedef cufftReal    Real;
typedef cufftComplex Complex;

int main(int argc, char *argv[])
{
    int sample_size = 1 << 20;      // 1,048,576
    const int batch_size = 1 << 9;  // 512

    Real    *p_sample;
    Complex *d_freq;
    cufftHandle plan_forward, plan_inverse;

    float forward_time_ms, inverse_time_ms;

    // create cuda event to measure the performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // create input / transform data memory space
    cudaMallocManaged((void**)&p_sample, sizeof(Real) * sample_size * batch_size);
    cudaMalloc((void**)&d_freq,   sizeof(Complex) * sample_size * batch_size);

    // create curand generator & set random seed
    curandGenerator_t curand_gen;
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 2019UL);
    curandGenerateUniform(curand_gen, p_sample, sample_size * batch_size);

    // 1D cufft setup
    int rank = 1;
    int stride_sample = 1, stride_freq = 1;
    int dist_sample = sample_size, dist_freq = sample_size / 2 + 1;
    int embed_sample[] = {0};
    int embed_freq[] = {0};
    cufftPlanMany(&plan_forward, rank, &sample_size,
                                 embed_sample, stride_sample, dist_sample, 
				                 embed_freq, stride_freq, dist_freq,
                                 CUFFT_R2C, batch_size);
    cufftPlanMany(&plan_inverse, rank, &sample_size,
                                 embed_freq, stride_freq, dist_freq, 
				                 embed_sample, stride_sample, dist_sample,
                                 CUFFT_C2R, batch_size);

    // executes FFT processes
    cudaEventRecord(start);
    cufftExecR2C(plan_forward, p_sample, d_freq);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&forward_time_ms, start, stop);

    // executes FFT processes (inverse transformation)
    cudaEventRecord(start);
    cufftExecC2R(plan_inverse, d_freq, p_sample);
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
