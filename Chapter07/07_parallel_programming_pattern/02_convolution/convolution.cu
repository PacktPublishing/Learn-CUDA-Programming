#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <helper_timer.h>
#include "helper_cuda.h"
#include <assert.h>

#define BLOCK_DIM   16
#define MAX_FILTER_LENGTH 128
#define RESULT_VERIFICATION 1   // change 1 if you want to verify the result

__global__ void
convolution_kernel_v1(float *d_output, float *d_input, float *d_filter, int num_row, int num_col, int filter_size)
{
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

    float result = 0.f;
    for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row)
    {
        for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
        {
            // Find the global position to apply the given filter
            int image_row = idx_y + filter_row;
            int image_col = idx_x + filter_col;

            float image_value = (image_row >= 0 && image_row < num_row && image_col >= 0 && image_col < num_col) ?
                                            d_input[image_row * num_col + image_col] : 0.f;
            float filter_value = d_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2];

            result += image_value * filter_value;
        }
    }

    d_output[idx_y * num_col + idx_x] = result;
}

__constant__ float c_filter[MAX_FILTER_LENGTH * MAX_FILTER_LENGTH];

__global__ void
convolution_kernel_v2(float *d_output, float *d_input, float *d_filter, int num_row, int num_col, int filter_size)
{
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

    float result = 0.f;
    for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row)
    {
        for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
        {
            int image_row = idx_y + filter_row;
            int image_col = idx_x + filter_col;

            float image_value = (image_row >= 0 && image_row < num_row && image_col >= 0 && image_col < num_col) ?
                                            d_input[image_row * num_col + image_col] : 0.f;
            float filter_value = c_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2];

            result += image_value * filter_value;
        }
    }

    d_output[idx_y * num_col + idx_x] = result;
}

__global__ void
convolution_kernel_v3(float *d_output, float *d_input, float *d_filter, int num_row, int num_col, int filter_size)
{
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    int pad_size = filter_size / 2;
    int tile_size = BLOCK_DIM + 2 * pad_size;

    extern __shared__ float s_input[];

    for (int row = 0; row <= tile_size / BLOCK_DIM; row++)
    {
        for (int col = 0; col <= tile_size / BLOCK_DIM; col++)
        {
            int idx_row = idx_y + BLOCK_DIM * row - pad_size;   // input data index row
            int idx_col = idx_x + BLOCK_DIM * col - pad_size;   // input data index column
            int fid_row = threadIdx.y + BLOCK_DIM * row; // filter index row
            int fid_col = threadIdx.x + BLOCK_DIM * col; // filter index column
            
            if (fid_row >= tile_size || fid_col >= tile_size)   continue;

            s_input[tile_size * fid_row + fid_col] = \
                (idx_row >= 0 && idx_row < num_row && idx_col >= 0 && idx_col < num_col) ? 
                    d_input[num_col * idx_row + idx_col] : 0.f;
        }
    }

    __syncthreads();

    /* Tile Debugging */
    // if (idx_x == BLOCK_DIM*1 && idx_y == BLOCK_DIM*1) 
    // {
    //     for (int row = 0; row < 2*pad_size + BLOCK_DIM; row++)
    //     {
    //         for (int col = 0; col < 2*pad_size + BLOCK_DIM; col++)
    //         {
    //             printf("%.0f ", s_input[tile_size * row + col]);
    //         }
    //         printf("\n");
    //     }
    // }

    float result = 0.f;
    for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row)
    {
        for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
        {
            // Find the global position to apply the given filter            
            int image_row = threadIdx.y + pad_size + filter_row;
            int image_col = threadIdx.x + pad_size + filter_col;

            float image_value  = s_input[tile_size * image_row + image_col];            
            float filter_value = c_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2];

            result += image_value * filter_value;
        }
    }

    d_output[idx_y * num_col + idx_x] = result;
}

void convolution_gpu(int version, float *d_output, float *d_input, float *d_filter, int num_row, int num_col, int filter_size)
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((num_col + BLOCK_DIM - 1) / BLOCK_DIM, (num_row + BLOCK_DIM - 1) / BLOCK_DIM);
    if (version == 1)
        convolution_kernel_v1<<<dimGrid, dimBlock>>>(d_output, d_input, d_filter, num_row, num_col, filter_size);
    else if (version == 2) 
        convolution_kernel_v2<<<dimGrid, dimBlock>>>(d_output, d_input, d_filter, num_row, num_col, filter_size);
    else // version == 3
    {
        int shared_mem_size = (2*filter_size+BLOCK_DIM) * (2*filter_size+BLOCK_DIM) * sizeof(float);
        convolution_kernel_v3<<<dimGrid, dimBlock, shared_mem_size, 0 >>>(d_output, d_input, d_filter, num_row, num_col, filter_size);
    }
    
    checkCudaErrors(cudaGetLastError());
}

void convolution_host(float *h_output, float *h_input, float *h_filter, int num_row, int num_col, int filter_size)
{
    //For every pixel in the image
    #pragma omp parallel 
    for (int row = 0; row < (int)num_row; ++row)
    {
        for (int col = 0; col < (int)num_col; ++col)
        {
            float result = 0.f;
            //For every value in the filter around the pixel (c, r)
            for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row)
            {
                for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
                {
                    // Find the global image position for this filter position
                    int image_row = row + filter_row;
                    int image_col = col + filter_col;

                    float image_value = (image_row >= 0 && image_row < num_row && image_col >= 0 && image_col < num_col) ?
                                            h_input[image_row * num_col + image_col] : 0.f;
                    float filter_value = h_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2];

                    result += image_value * filter_value;
                }
            }

            h_output[row * num_col + col] = result;
        }
    }
}


/* Generates Bi-symetric Gaussian Filter */
void generate_filter(float *h_filter, int filter_size)
{
    float blur_kernel_sigma = 2.;

    float sum_filter = 0.f; //for normalization
    for (int row = -filter_size / 2; row <= filter_size / 2; row++)
    {
        for (int col = -filter_size / 2; col <= filter_size / 2; col++)
        {
            float filterValue = expf(-(float)(col * col + row * row) / (2.f * blur_kernel_sigma * blur_kernel_sigma));
            h_filter[(row + filter_size / 2) * filter_size + col + filter_size / 2] = filterValue;
            sum_filter += filterValue;
        }
    }

    // normalization
    float normalizationFactor = 1.f / sum_filter;
    for (int row = -filter_size / 2; row <= filter_size / 2; row++)
        for (int col = -filter_size / 2; col <= filter_size / 2; col++)
            h_filter[(row + filter_size / 2) * filter_size + col + filter_size / 2] *= normalizationFactor;
}

void generate_data(float *h_buffer, int num_row, int num_col)
{
    for (int row = 0; row < num_row; row++) {
        for (int col = 0; col < num_col; col++) {
            // h_buffer[row * num_col + col] = float(rand() & 0xFFFFFF) / RAND_MAX;
            h_buffer[row * num_col + col] = 1.f;
        }
    }
}

bool value_test(float *a, float *b, int length)
{
    float epsilon = 0.000001;
    bool result = true;
    for (int i = 0; i < length; i++)
        if (abs(a[i] - b[i]) >= epsilon)
            result = false;
    return result;
}

int main()
{
    int num_row = 2048;
    int num_col = 2048;
    int filter_size = 9;
    int buf_size = num_row * num_col * sizeof(float);

    float *h_input, *d_input;
    float *h_output_host, *h_output_gpu, *d_output;
    float *h_filter, *d_filter;

    float elapsed_time_gpu;

    // initialize timer
    StopWatchInterface *timer_host, *timer_gpu;
    sdkCreateTimer(&timer_host);
    sdkCreateTimer(&timer_gpu);

    srand(2019);

    // allocate host memories
    h_input = (float *)malloc(buf_size);
    h_output_host = (float *)malloc(buf_size);
    h_output_gpu = (float *)malloc(buf_size);
    h_filter = (float *)malloc(filter_size * filter_size * sizeof(float));

    // allocate gpu memories
    cudaMalloc((void **)&d_input, buf_size);
    cudaMalloc((void **)&d_output, buf_size);
    cudaMalloc((void **)&d_filter, filter_size * filter_size * sizeof(float));

    // generate data
    generate_data(h_input, num_row, num_col);
    generate_filter(h_filter, filter_size);

    // copy input date to gpu
    cudaMemcpy(d_input, h_input, buf_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_filter, h_filter, filter_size * filter_size * sizeof(float));

    // processing in GPU
    sdkStartTimer(&timer_gpu);
    cudaProfilerStart();
    convolution_gpu(1, d_output, d_input, d_filter, num_row, num_col, filter_size);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer_gpu);
    elapsed_time_gpu = sdkGetTimerValue(&timer_gpu);
    printf("Processing Time (1) -> GPU: %.2f ms\n", elapsed_time_gpu);

    // processing in GPU
    sdkResetTimer(&timer_gpu);
    sdkStartTimer(&timer_gpu);
    convolution_gpu(2, d_output, d_input, d_filter, num_row, num_col, filter_size);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer_gpu);
    elapsed_time_gpu = sdkGetTimerValue(&timer_gpu);
    printf("Processing Time (2) -> GPU: %.2f ms\n", elapsed_time_gpu);

    // processing in GPU (3)
    sdkResetTimer(&timer_gpu);
    sdkStartTimer(&timer_gpu);
    convolution_gpu(3, d_output, d_input, d_filter, num_row, num_col, filter_size);
    cudaDeviceSynchronize();
    sdkStopTimer(&timer_gpu);
    cudaProfilerStop();
    elapsed_time_gpu = sdkGetTimerValue(&timer_gpu);
    printf("Processing Time (3) -> GPU: %.2f ms\n", elapsed_time_gpu);

#if (RESULT_VERIFICATION)
    // processing in CPU
    sdkStartTimer(&timer_host);
    convolution_host(h_output_host, h_input, h_filter, num_row, num_col, filter_size);
    sdkStopTimer(&timer_host);

    float elapsed_time_host = sdkGetTimerValue(&timer_host);
    printf("Processing Time -> Host: %.2f ms\n", elapsed_time_host);

    // compare the result
    cudaMemcpy(h_output_gpu, d_output, buf_size, cudaMemcpyDeviceToHost);
    if (value_test(h_output_host, h_output_gpu, num_row * num_col))
        printf("SUCCESS!!\n");
    else
        printf("Error\n");
#endif

    // finalize
    free(h_input);
    free(h_output_host);
    free(h_output_gpu);
    free(h_filter);

    return 0;
}

