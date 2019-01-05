#include <iostream>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <cuda_fp16.h>

typedef cufftReal   Real;
typedef half2       Complex;

int main(int argc, char *argv[])
{
    long long sample_size = 1 << 20;  // 1,048,576
    const int batch_size = 1 << 9;   // 512

    cufftHandle plan_forward, plan_inverse;
    Real    *h_sample, *d_sample;
    Complex *d_freq;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // create host buffer as input data
    h_sample = (Real*) new Real[sample_size * batch_size];

    // create signal and filter memory
    cudaMalloc((void**)&d_sample, sample_size * sizeof(Real) * batch_size);
    cudaMalloc((void**)&d_freq, sample_size * sizeof(Complex) * batch_size);

    // 1D cufft setup
    cufftCreate(&plan_forward);
    cufftCreate(&plan_inverse);

    int rank = 1;
    int stride_sample = 1, stride_freq = 1;
    long long int dist_sample = sample_size, dist_freq = sample_size / 2 + 1;
    long long embed_sample[] = {0};
    long long embed_freq[] = {0};
    size_t workSize = 0;
    cufftXtMakePlanMany(plan_forward, 
        rank, &sample_size, 
        embed_sample, stride_sample, dist_sample, CUDA_C_16F, 
        embed_freq, stride_freq, dist_freq, CUDA_C_16F, 
        batch_size, &workSize, CUDA_C_16F);
    cufftXtMakePlanMany(plan_inverse,
    	rank, &sample_size,
        embed_freq, stride_freq, dist_freq, CUDA_C_16F,
        embed_sample, stride_sample, dist_sample, CUDA_C_16F,
        batch_size, &workSize, CUDA_C_16F);

    // copy input data from the host to the device
    cudaMemcpy(d_sample, h_sample, sample_size * sizeof(Real) * batch_size, cudaMemcpyHostToDevice);

    // record start event
    cudaEventRecord(start);

    // executes FFT processes
    cufftXtExec(plan_forward, d_sample, d_freq, CUFFT_FORWARD);

    // record stop event
    cudaEventRecord(stop);

    // executes FFT processes (inverse transformation)
    cufftXtExec(plan_inverse, d_freq, d_sample, CUFFT_INVERSE);

    // copy input data from the host to the device
    cudaMemcpy(h_sample, d_sample, sample_size * sizeof(Real) * batch_size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float operation_time = 0.f;
    cudaEventElapsedTime(&operation_time, start, stop);
    printf("FFT operation time for %lld samples with %d batch: %f ms\n", sample_size, batch_size, operation_time);

    //delete[] workSize;

    // delete cuda events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // deletes CUFFT plan_forward
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);

    // terminates memories
    delete[] h_sample;
    cudaFree(d_sample);

    return 0;
}
