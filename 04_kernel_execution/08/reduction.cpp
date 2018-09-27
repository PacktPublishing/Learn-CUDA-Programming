#include <stdio.h>
#include <stdlib.h>

// cuda runtime
#include <cuda_runtime.h>
#include <helper_timer.h>

#include "reduction.h"

void run_benchmark(int (*reduce)(float*, float*, int, int), 
                   float *d_outPtr, float *d_inPtr, int size);
void init_input(float* data, int size);
float get_cpu_result(float *data, int size);


// 
int check_cooperative_launch_support()
{
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, 0);
    if (deviceProp.cooperativeLaunch == 0)
        return 0;

    return 1;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main(int argc, char *argv[])
{
    float *h_inPtr;
    float *d_inPtr, *d_outPtr;

    unsigned int size = 1 << 24;

    float result_host, result_gpu;
    int mode = 0;

    // check device availability
    if (check_cooperative_launch_support() == 0)
    {
        puts("Target GPU does not support Cooperative Kernel Launch. Exit.");
        exit(EXIT_FAILURE);
    }

    srand(2019);

    // Allocate memory
    h_inPtr = (float*)malloc(size * sizeof(float));
    
    // Data initialization with random values
    init_input(h_inPtr, size);

    // Prepare GPU resource
    cudaMalloc((void**)& d_inPtr, size * sizeof(float));
    cudaMalloc((void**)&d_outPtr, size * sizeof(float));

    cudaMemcpy(d_inPtr, h_inPtr, size * sizeof(float), cudaMemcpyHostToDevice);

    // Get reduction result from GPU (reduction 0)
    run_benchmark(reduction_grid_sync, d_outPtr, d_inPtr, size);
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

void
run_benchmark(int (*reduce)(float*, float*, int, int), 
              float *d_outPtr, float *d_inPtr, int size)
{
    int num_threads = 256;
    int test_iter = 100;

    // warm-up
    reduce(d_outPtr, d_inPtr, size, num_threads);

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    ////////
    // Operation body
    ////////
    for (int i = 0; i < test_iter; i++) {
        reduce(d_outPtr, d_inPtr, size, num_threads);
    }

    // getting elapsed time
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    // Compute and print the performance
    float elapsed_time_msed = sdkGetTimerValue(&timer) / (float)test_iter;
    float bandwidth = size * sizeof(float) / elapsed_time_msed / 1e6;
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    sdkDeleteTimer(&timer);
}

void
init_input(float *data, int size)
{
    for (int i = 0; i < size; i++) {
        // Keep the numbers small so we don't get truncation error in the sum
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

float
get_cpu_result(float *data, int size)
{
    double result = 0.f;
    for (int i = 0; i < size; i++)
        result += data[i];

    return (float)result;
}