#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include <curand.h>
#include "helper_cuda.h"
#include "helper.cuh"

typedef cufftReal    Real;
typedef cufftComplex Complex;

int main(int argc, char *argv[])
{
    long long sample_size = 1 << 20;      // 1,048,576
    const int batch_size = 1 << 9;  // 512
    int n_gpu = 2;

    cufftHandle cufft_plan;
    Complex *d_input;
    Complex *h_input, *h_output;

    float forward_time_ms, inverse_time_ms;

    // create cuda event to measure the performance
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    h_input = (Complex*) new Complex[sample_size * batch_size];
    h_output= (Complex*) new Complex[sample_size * batch_size];

    // create curand generator & set random seed
    curandGenerator_t curand_gen;
    cudaMalloc((void**)&d_input, sizeof(Complex) * sample_size * batch_size);
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 2019UL);
    op::curand(curand_gen, d_input, sample_size * batch_size);
    cudaMemcpy(h_input, d_input, sizeof(Complex) * sample_size * batch_size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);

    // 1. create cufft empty plan
    cufftCreate(&cufft_plan);

    // 2. set multi-gpu
    int *devices = (int*) new int[n_gpu];
    for (int i = 0; i < n_gpu; i++)
        devices[i] = i;
    cufftXtSetGPUs(cufft_plan, n_gpu, devices);

    // 3. create teh plan
    size_t *workSize = (size_t*) new size_t[n_gpu];
    cufftXtMakePlanMany(cufft_plan, 1, &sample_size,
                        nullptr, 1, 1, CUDA_C_32F,
                        nullptr, 1, 1, CUDA_C_32F,
                        batch_size, workSize, CUDA_C_32F);

    // 4. allocate multi-gpu memory space and copy data from the host
    cudaLibXtDesc *d_sample;
    checkCudaErrors(cufftXtMalloc(cufft_plan, &d_sample, CUFFT_XT_FORMAT_INPLACE));
    checkCudaErrors(cufftXtMemcpy(cufft_plan, d_sample, h_input, CUFFT_COPY_HOST_TO_DEVICE));

    // 5. executes FFT processes
    cudaEventRecord(start);
    checkCudaErrors(cufftXtExecDescriptor(cufft_plan, d_sample, d_sample, CUFFT_FORWARD));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkCudaErrors(cudaEventElapsedTime(&forward_time_ms, start, stop));

    // 6. executes FFT processes (inverse transformation)
    cudaEventRecord(start);
    cufftXtExecDescriptor(cufft_plan, d_sample, d_sample, CUFFT_INVERSE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&inverse_time_ms, start, stop);

    // 7. copy the result to the host
    cufftXtMemcpy(cufft_plan, h_output, d_sample, CUFFT_COPY_DEVICE_TO_HOST);

    // print elapsed time
    std::cout << "FFT operation time for " << sample_size << " elements with " << batch_size << " batch.." << std::endl;
    std::cout << "Forward (ms): " << forward_time_ms << std::endl;
    std::cout << "Inverse (ms): " << inverse_time_ms << std::endl;

    // terminates used resources
    curandDestroyGenerator(curand_gen);

    // deletes CUFFT plan_forward
    cufftDestroy(cufft_plan);

    // terminates memories
    cufftXtFree(d_sample);
    
    // delete cuda events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete [] h_input;
    delete [] h_output;

    return 0;
}
