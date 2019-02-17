#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "../03_scan/scan_v2.cu"
#include "../03_scan/utils.h"

#define FILTER_MASK 0.f
#define GRID_DIM    1       // this implementation covers 1 thread block only

#define BLOCK_DIM 16

void generate_data(float *ptr, int length);

// predicate
// mark elements which will be scattered
__global__ void
predicate_kernel(float *d_predicates, float *d_input, int length)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= length) return;

    d_predicates[idx] = d_input[idx] > FILTER_MASK;
}

// scan

// address
__global__ void
pack_kernel(float *d_output, float *d_input, float *d_predicates, float *d_scanned, int length)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= length) return;

    if (d_predicates[idx] != 0.f)
    {
        // address
        int address = d_scanned[idx] - 1;

        // scatter
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
        if (h_input[i] > FILTER_MASK)
        {
            h_output[idx_output] = h_input[i];
            idx_output++;
        }
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

    if (length <= DEBUG_OUTPUT_NUM)
        print_val(h_input, length, "input   ::");

    // predicates
    predicate_kernel<<< GRID_DIM, BLOCK_DIM >>>(d_predicates, d_input, length);

    // debug
    cudaMemcpy(h_output_gpu, d_predicates, sizeof(float) * length, cudaMemcpyDeviceToHost);

    // scan
    scan_v2(d_scanned, d_predicates, length);

    // addressing & scatter (pack)
    pack_kernel<<< GRID_DIM, BLOCK_DIM >>>(d_output, d_input, d_predicates, d_scanned, length);
    cudaDeviceSynchronize();

    // validation the result (pack)
    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * length, cudaMemcpyDeviceToHost);
    pack_host(h_output_host, h_input, length);
    if (validation(h_output_host, h_output_gpu, length))
        printf("SUCCESS!!\n");
    else
        printf("Something wrong..\n");

    // debug
    if (length <= DEBUG_OUTPUT_NUM)
        print_val(h_output_host, length, "pack[host]::");
    if (length <= DEBUG_OUTPUT_NUM)
        print_val(h_output_gpu, length, "pack[gpu] ::");

    // split
    cudaMemset(d_input, 0, sizeof(float) * length);
    split_kernel<<<GRID_DIM, BLOCK_DIM>>>(d_input, d_output, d_predicates, d_scanned, length);
    cudaDeviceSynchronize();

    // validation the result (split)
    cudaMemcpy(h_output_gpu, d_input, sizeof(float) * length, cudaMemcpyDeviceToHost);
    if (validation(h_output_host, h_output_gpu, length))
        printf("SUCCESS!!\n");
    else
        printf("Something wrong..\n");

    if (length <= DEBUG_OUTPUT_NUM)
        print_val(h_output_gpu, length, "split[gpu]");

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


