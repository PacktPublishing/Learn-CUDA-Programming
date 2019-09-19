#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "scan.h"
#include "utils.h"

#define SCAN_VERSION    1   // 1: naîve scan, 2: Blelloch scan

void scan_host(float *h_output, float *h_input, int length, int version);

int main()
{
    srand(2019);
    float *h_input, *h_output_host, *h_output_gpu;
    float *d_input, *d_output;
    int length = BLOCK_DIM * 2;

    // host memory allocation
    h_input       = (float *)malloc(sizeof(float) * length);
    h_output_host = (float *)malloc(sizeof(float) * length);
    h_output_gpu  = (float *)malloc(sizeof(float) * length);

    // devide memory allocation
    cudaMalloc((void**)&d_input,  sizeof(float) * length);
    cudaMalloc((void**)&d_output, sizeof(float) * length);

    // generate input data
    generate_data(h_input, length);
    print_val(h_input, DEBUG_OUTPUT_NUM, "input         ::");

    // naïve scan
    scan_host(h_output_host, h_input, length, 1);
    print_val(h_output_host, DEBUG_OUTPUT_NUM, "result[cpu]   ::");

    cudaMemcpy(d_input, h_input, sizeof(float) * length, cudaMemcpyHostToDevice);
    scan_v1(d_output, d_input, length);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * length, cudaMemcpyDeviceToHost);
    print_val(h_output_gpu, DEBUG_OUTPUT_NUM, "result[gpu_v1]::");
    if (validation(h_output_host, h_output_gpu, length))
        printf("SUCCESS!!\n");

    // belloch scan
    scan_host(h_output_host, h_input, length, 2);
    print_val(h_output_host, DEBUG_OUTPUT_NUM, "result[cpu]   ::");

    cudaMemcpy(d_input, h_input, sizeof(float) * length, cudaMemcpyHostToDevice);
    scan_v2(d_output, d_input, length);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * length, cudaMemcpyDeviceToHost);
    print_val(h_output_gpu, DEBUG_OUTPUT_NUM, "result[gpu_v2]::");
    if (validation(h_output_host, h_output_gpu, length))
        printf("SUCCESS!!\n");

    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // free host memory
    free(h_input);
    free(h_output_host);
    free(h_output_gpu);

    return 0;
}

void scan_host(float *h_output, float *h_input, int length, int version)
{
    bool debug = false;

    if (debug)
    {
        printf("input ]]\t");
        for (int i = 0; i < length; i++) {
            printf("%3.f  ", h_input[i]);
        }
        printf("\n");
    }

    if (version == 1)
    {
        for (int i = 0; i < length; i++)
            for (int j = 0; j < length; j++)
                if (i - j >= 0)
                    h_output[i] += h_input[i - j];
    }
    else
    {
        for (int i = 0; i < length; i++)
        {
            h_output[i] = h_input[i];
        }

        int offset = 1;
        while (offset < length)
        {
            for (int i = 0; i < length; i++)
            {
                int idx_a = offset * (2 * i + 1) - 1;
                int idx_b = offset * (2 * i + 2) - 1;

                if (idx_a >= 0 && idx_b < length)
                    h_output[idx_b] += h_output[idx_a];
            }
            offset <<= 1;
        }

        offset >>= 1;
        while (offset > 0)
        {
            for (int i = 0; i < length; i++)
            {
                int idx_a = offset * (2 * i + 2) - 1;
                int idx_b = offset * (2 * i + 3) - 1;

                if (idx_a >= 0 && idx_b < length)
                    h_output[idx_b] += h_output[idx_a];
            }
            offset >>= 1;
        }
    }

    if (debug)
    {
        printf("output ]]\t");
        for (int i = 0; i < length; i++)
        {
            printf("%3.f  ", h_output[i]);
        }
        printf("\n");
    }
}
