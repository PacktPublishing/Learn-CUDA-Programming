#include <iostream>
#include <cuda_runtime.h>
#include <cufft.h>

typedef cufftReal    Real;
typedef cufftComplex Complex;

void GenerateSample(float* buffer, size_t length)
{
    for (int i = 0; i < length; i++) 
        buffer[i] = rand() & 0xFF / RAND_MAX;
}

int main(int argc, char *argv[])
{
    int sample_size = 1 << 20;      // 1,048,576
    const int batch_size = 1 << 9;  // 512

    cufftHandle plan_forward, plan_inverse;
    Real    *h_sample, *d_sample;
    Complex *d_freq;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // create host buffer as input data
    h_sample = (Real*) new Real[sample_size * batch_size];

    // Set input data
    srand(2019);
    for (int i = 0; i < batch_size; i++)
        GenerateSample(&h_sample[i * sample_size], sample_size);

    // create signal and filter memory
    cudaMalloc((void**)&d_sample, sample_size * sizeof(Real) * batch_size);
    cudaMalloc((void**)&d_freq,   sample_size * sizeof(Complex) * batch_size);

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

    // copy input data from the host to the device
    cudaMemcpy(d_sample, h_sample, sample_size * sizeof(Real) * batch_size, cudaMemcpyHostToDevice);

    // record start event
    cudaEventRecord(start);

    // executes FFT processes
    cufftExecR2C(plan_forward, d_sample, d_freq);

    // record stop event
    cudaEventRecord(stop);

    // executes FFT processes (inverse transformation)
    cufftExecC2R(plan_inverse, d_freq, d_sample);

    // copy the result back to the host
    cudaMemcpy(h_sample, d_sample, sample_size * sizeof(Real) * batch_size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float operation_time = 0.f;
    cudaEventElapsedTime(&operation_time, start, stop);
    printf("FFT operation time for %d samples with %d batch: %f ms\n", sample_size, batch_size, operation_time);

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
