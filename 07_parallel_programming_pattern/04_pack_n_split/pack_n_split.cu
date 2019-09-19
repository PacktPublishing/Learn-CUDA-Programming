#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "../03_scan/utils.h"
#include "../03_scan/scan_v2.cu"

#define FLT_ZERO 0.f
#define GRID_DIM    1       // this implementation covers only 1 thread block's operation

// #define BLOCK_DIM 16

void generate_data(float *ptr, int length);

// predicate
// mark elements which will be scattered
__global__ void
predicate_kernel(float *d_predicates, float *d_input, int length)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= length) return;

    d_predicates[idx] = d_input[idx] > FLT_ZERO;
}

// scan
/* We will use the previous implementation (#include "../03_scan/scan_v2.cu") */

// address and gather
__global__ void
pack_kernel(float *d_output, float *d_input, float *d_predicates, float *d_scanned, int length)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= length) return;

    if (d_predicates[idx] != 0.f)
    {
        // address
        int address = d_scanned[idx] - 1;

        // gather
        d_output[address] = d_input[idx];
    }
}

__global__ void
split_kernel(float *d_output, float *d_input, float *d_predicates, float *d_scanned, int length)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= length) return;

    if (d_predicates[idx] != 0.f)
    {
        // address
        int address = d_scanned[idx] - 1;

        // split
        d_output[idx] = d_input[address];
    }
}

// pack_host : evaluation purpose
void pack_host(float *h_output, float *h_input, int length)
{
    int idx_output = 0;
    for (int i = 0; i < length; i++)
    {
        if (h_input[i] > FLT_ZERO)
        {
            h_output[idx_output] = h_input[i];
            idx_output++;
        }
    }
}

// split_host: pseudo implementation for evaluation purpose
void split_host(float *h_output, float *h_input, int length)
{
    for (int i = 0; i < length; i++)
    {
        if (h_input[i] >= 0.f)
            h_output[i] = h_input[i];
        else
            h_output[i] = 0.f;
    }
}

int main()
{
    float *h_input, *h_output_host, *h_output_gpu;
    float *d_input, *d_output;
    float *d_predicates, *d_scanned; // for temporarly purpose operation
    float length = BLOCK_DIM;

    srand(2019);

    // allocate host memory
    h_input = (float *)malloc(sizeof(float) * length);
    h_output_host = (float *)malloc(sizeof(float) * length);
    h_output_gpu = (float *)malloc(sizeof(float) * length);

    // allocate device memory
    cudaMalloc((void**)&d_input, sizeof(float) * length);
    cudaMalloc((void**)&d_output, sizeof(float) * length);
    cudaMalloc((void**)&d_predicates, sizeof(float) * length);
    cudaMalloc((void**)&d_scanned, sizeof(float) * length);

    // generate input data
    generate_data(h_input, length);
    cudaMemcpy(d_input, h_input, sizeof(float) * length, cudaMemcpyHostToDevice);

    print_val(h_input, DEBUG_OUTPUT_NUM, "input    ::");

    cudaProfilerStart();
    /********************************
     * Pack                         *
     ********************************/
    // predicates
    predicate_kernel<<< GRID_DIM, BLOCK_DIM >>>(d_predicates, d_input, length);

    // scan
    scan_v2(d_scanned, d_predicates, length);

    // addressing & gather (pack)
    pack_kernel<<< GRID_DIM, BLOCK_DIM >>>(d_output, d_input, d_predicates, d_scanned, length);
    cudaDeviceSynchronize();

    // validation the result (compack)
    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * length, cudaMemcpyDeviceToHost);
    pack_host(h_output_host, h_input, length);

    print_val(h_output_host, DEBUG_OUTPUT_NUM, "pack[cpu]::");
    print_val(h_output_gpu, DEBUG_OUTPUT_NUM, "pack[gpu]::");

    if (validation(h_output_host, h_output_gpu, length))
        printf("SUCCESS!!\n");
    else
        printf("Something wrong..\n");

    /********************************
     * Split                        *
     ********************************/
    cudaMemcpy(d_input, d_output, sizeof(float) * length, cudaMemcpyDeviceToDevice);
    cudaMemset(d_output, 0, sizeof(float) * length);
    split_kernel<<<GRID_DIM, BLOCK_DIM>>>(d_output, d_input, d_predicates, d_scanned, length);
    cudaDeviceSynchronize();
    cudaProfilerStop();

    // validation the result (split)
    cudaMemcpy(h_output_gpu,  d_output, sizeof(float) * length, cudaMemcpyDeviceToHost);
    split_host(h_output_host, h_input, length); // notice: we just generate desired output for the evaluation purpose

    print_val(h_output_gpu, DEBUG_OUTPUT_NUM, "split[gpu]");
    if (validation(h_output_host, h_output_gpu, length))
        printf("SUCCESS!!\n");
    else
        printf("Something wrong..\n");

    // finalize
    cudaFree(d_predicates);
    cudaFree(d_scanned);
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output_gpu);
    free(h_output_host);
    free(h_input);

    return 0;
}


