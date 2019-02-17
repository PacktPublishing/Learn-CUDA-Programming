#include "scan.h"

__global__ void
scan_v2_kernel(float *d_output, float *d_input, int length)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float s_buffer[];

    int offset = 1;

    s_buffer[threadIdx.x] = d_input[idx];
    s_buffer[threadIdx.x + BLOCK_DIM] = d_input[idx + BLOCK_DIM];

    // printf("[%d, %3.f]\t", tid, s_buffer[tid]);
    // printf("[%d, %3.f]\t", tid, s_buffer[tid + BLOCK_DIM]);
    // if (tid == 0)
    //     printf("\n");

    while (offset < length)
    {
        __syncthreads();

        int idx_a = offset * (2 * tid + 1) - 1;
        int idx_b = offset * (2 * tid + 2) - 1;

        if (idx_a >= 0 && idx_b < 2 * BLOCK_DIM)
        {
            // printf("<< %d, %d >>\n", idx_a, idx_b);
            s_buffer[idx_b] += s_buffer[idx_a];
        }

        offset <<= 1;
    }

    // printf("[%d, %3.f]\t", tid, s_buffer[tid]);
    // printf("[%d, %3.f]\t", tid, s_buffer[tid + BLOCK_DIM]);
    // if (tid == 0)
    //     printf("\n");

    offset >>= 1;
    while (offset > 0)
    {
        __syncthreads();

        int idx_a = offset * (2 * tid + 2) - 1;
        int idx_b = offset * (2 * tid + 3) - 1;

        if (idx_a >= 0 && idx_b < 2 * BLOCK_DIM)
        {
            s_buffer[idx_b] += s_buffer[idx_a];
            // printf("<< %d, %d >>\n", idx_a, idx_b);
        }

        offset >>= 1;
    }
    __syncthreads();

    // printf("[%d, %3.f]\t", tid, s_buffer[tid]);
    // printf("[%d, %3.f]\t", tid, s_buffer[tid + BLOCK_DIM]);

    d_output[idx] = s_buffer[tid];
    d_output[idx + BLOCK_DIM] = s_buffer[tid + BLOCK_DIM];
}

void scan_v2(float *d_output, float *d_input, int length)
{
    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid((length + (2 * BLOCK_DIM) - 1) / (2 * BLOCK_DIM));
    scan_v2_kernel<<<dimGrid, dimBlock, sizeof(float) * BLOCK_DIM * 2>>>(d_output, d_input, length);
    cudaDeviceSynchronize();
}