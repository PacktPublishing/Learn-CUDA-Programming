#include <stdio.h>
#include <stdlib.h>

// cuda runtime
#include <cuda_runtime.h>

#include "reduction.h"

void run_benchmark(void (*reduce)(float *, float *, int, int, int),
                   float *d_outPtr, float *d_inPtr, int size);
void init_input(float *data, int size);
float get_cpu_result(float *data, int size);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
    float *h_inPtr;
    float *d_inPtr, *d_outPtr;

    unsigned int size = 1 << 24;

    float result_host, result_gpu;
    int mode = -1;

    if (argc > 1)
    {
        mode = atoi(argv[1]);
        if (mode < 0 || mode > 2)
        {
            puts("Invalid reduction request!! 0-2 are avaiable.");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        puts("Please put operation option!! 0-2 are avaiable.");
        exit(EXIT_FAILURE);
    }

    srand(2019);

    // Allocate memory
    h_inPtr = (float *)malloc(size * sizeof(float));

    // Data initialization with random values
    init_input(h_inPtr, size);

    // Prepare GPU resource
    cudaMalloc((void **)&d_inPtr, size * sizeof(float));
    cudaMalloc((void **)&d_outPtr, size * sizeof(float));

    cudaMemcpy(d_inPtr, h_inPtr, size * sizeof(float), cudaMemcpyHostToDevice);

    // Get reduction result from GPU
    switch (mode) {
        case 0:
            run_benchmark(fault_reduction, d_outPtr, d_inPtr, size);
            break;
        case 1:
            run_benchmark(atomic_reduction, d_outPtr, d_inPtr, size);
            break;
    }
    cudaMemcpy(&result_gpu, &d_outPtr[0], sizeof(float), cudaMemcpyDeviceToHost);

    // Get reduction result from GPU

    // Get all sum from CPU
    result_host = get_cpu_result(h_inPtr, size);
    printf("host: %f, device %f\n", result_host, result_gpu);

    // Terminates memory
    cudaFree(d_outPtr);
    cudaFree(d_inPtr);
    free(h_inPtr);

    return 0;
}

void run_reduction(void (*reduce)(float *, float *, int, int, int),
                   float *d_outPtr, float *d_inPtr, int size, int n_threads)
{
    int n_blocks = (size + n_threads - 1) / n_threads;
    reduce(d_outPtr, d_inPtr, size, n_blocks, n_threads);
    cudaDeviceSynchronize();
}

void run_benchmark(void (*reduce)(float *, float *, int, int, int),
                   float *d_outPtr, float *d_inPtr, int size)
{
    cudaEvent_t start, stop;
    int num_threads = 256;
    int test_iter = 100;

    // Allocate CUDA events that we'll use for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm-up
    reduce(d_outPtr, d_inPtr, size, size / num_threads, num_threads);

    // Record the start event
    cudaEventRecord(start, NULL);

    ////////
    // Operation body
    ////////
    for (int i = 0; i < test_iter; i++)
    {
        cudaMemcpy(d_outPtr, d_inPtr, size * sizeof(float), cudaMemcpyDeviceToDevice);
        run_reduction(reduce, d_outPtr, d_outPtr, size, num_threads);
    }

    // Record the stop event
    cudaEventRecord(stop, NULL);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    msecTotal /= (float)test_iter;

    // Compute and print the performance
    float bandwidth = size * sizeof(float) / msecTotal / 1e6;

    printf(
        "Time= %.3f msec, bandwidth= %f GB/s\n",
        msecTotal, bandwidth / msecTotal);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void init_input(float *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        // Keep the numbers small so we don't get truncation error in the sum
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

float get_cpu_result(float *data, int size)
{
    double result = 0.f;
    for (int i = 0; i < size; i++)
        result += data[i];

    return (float)result;
}