#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "scan.h"
#include "scan_v1.cu"
#include "scan_v2.cu"

void scan_host(float *h_output, float *h_input, int length)
{
    bool debug = false;
    int version = 1;
    float temp = 0.f;

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
        {
            temp += h_input[i];
            h_output[i] = temp;
        }
    }
    else
    {
        for (int offset = 1; offset < length; offset <<= 1)
        {
            int stride = 2 * offset;
            for (int i = 0; i < length; i++)
            {
                int idx_a = stride * i + stride - offset - 1;
                int idx_b = stride * i + stride - 1;

                idx_a = offset * (2 * i + 1) - 1;
                idx_b = offset * (2 * i + 2) - 1;

                if (idx_a >= 0 && idx_b < length)
                    h_output[idx_b] += h_input[idx_a];
            }
        }

        for (int offset = length >> 1; offset > 0; offset >>= 1)
        {
            int stride = 2 * offset;
            for (int i = 0; i < length; i++)
            {
                int idx_a = stride * i + stride - 1;
                int idx_b = stride * i + stride - 1 + offset;

                idx_a = offset * (2 * i + 2) - 1;
                idx_b = offset * (2 * i + 3) - 1;

                if (idx_a >= 0 && idx_b < length)
                    h_output[idx_b] += h_output[idx_a];
            }
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

int main()
{
    srand(2019);
    float *h_input, *h_output_host, *h_output_gpu;
    float *d_input, *d_output;
    int length = NUM_ITEM;

    // host memory allocation
    h_input = (float *)malloc(sizeof(float) * length);
    h_output_host   = (float *)malloc(sizeof(float) * length);
    h_output_gpu = (float *)malloc(sizeof(float) * length);

    // devide memory allocation
    cudaMalloc((void**)&d_input, sizeof(float) * length);
    cudaMalloc((void**)&d_output, sizeof(float) * length);

    // generate input data
    generate_data(h_input, length);

    if (length <= DEBUG_OUTPUT_NUM)
        print_val(h_input, length, "input      ::");

    // serial scan (host)
    scan_host(h_output_host, h_input, length);

    // naïve scan
    // cudaMemcpy(d_input, h_input, sizeof(float) * length, cudaMemcpyHostToDevice);
    // scan_v1(d_output, d_input, length);
    // cudaDeviceSynchronize();
    // cudaMemcpy(h_output_gpu, d_output, sizeof(float) * length, cudaMemcpyDeviceToHost);

    // belloch scan
    cudaMemcpy(d_input, h_input, sizeof(float) * length, cudaMemcpyHostToDevice);
    scan_v2(d_output, d_input, length);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_output, sizeof(float) * length, cudaMemcpyDeviceToHost);

    // compare the result
    if (validation(h_output_host, h_output_gpu, length))
        printf("SUCCESS!!\n");
    else
        printf("Error\n");
    // printf("h:%f, d:%f\n", h_output_host[NUM_ITEM-1], h_output_gpu[NUM_ITEM-1]);

    if (length <= DEBUG_OUTPUT_NUM)
        print_val(h_output_gpu, length, "result[gpu]::");

    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // free host memory
    free(h_input);
    free(h_output_host);
    free(h_output_gpu);

    return 0;
}

