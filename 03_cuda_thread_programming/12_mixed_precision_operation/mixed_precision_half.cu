
// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__ARITHMETIC.html
// FP16 types and intrinsics
#include <cuda_fp16.h>
#include <stdlib.h>
#include <helper_timer.h>
#include <helper_math.h>
#include <cooperative_groups.h>
#include <cstdio>
#include "util.cuh"

using namespace cooperative_groups;

__global__ void hfma_kernel(half *d_x, half *d_y, float *d_z, int height, int width)
{
    grid_group grid = this_grid();
    thread_block block = this_thread_block();

    int idx_x = block.group_dim().x * block.group_index().x + block.thread_index().x;
    int idx_y = block.group_dim().y * block.group_index().y + block.thread_index().y;

#if __CUDA_ARCH__ >= 530
    int half_size = width / 2;
    half2 *dual_x = (half2 *)d_x;
    half2 *dual_y = (half2 *)d_y;

    float sum = 0.f;
    for (int i = 0; i < half_size; i++) {
        float2 temp = __half22float2(__hmul2(dual_y[idx_y * half_size + i], dual_x[idx_y * half_size + i]));
        sum += temp.x + temp.y;
    }
    d_z[idx_y * width + idx_x] = sum;
#else
    float sum = 0.f;
    for (int i = 0; i < width; i++)
        sum += __half2float(d_x[idx_y * width + i]) * __half2float(d_y[idx_y * width + i]);
    
    d_z[idx_y * width + idx_x] = sum;
#endif
}

__global__ void float2half_kernel(half *d_out, float *d_in)
{
    grid_group grid = this_grid();

    d_out[grid.thread_rank()] = __float2half(d_in[grid.thread_rank()]);
}

int main()
{
    CBuffer<float> X, Y, Z;
    int height = 1 << 12;
    int width = 1 << 12;
    int size = height * width;
    int n_iteration = 100;

    srand(2019);

    // initialize host buffers
    X.init(size, true);
    Y.init(size, true);
    Z.init(size, false);

    // initalize gpu buffers
    X.cuda(true);
    Y.cuda(true);
    Z.cuda(true);

    // initialize half input data
    half *d_half_x, *d_half_y;
    cudaMalloc((void**)&d_half_x, size * sizeof(half));
    cudaMalloc((void**)&d_half_y, size * sizeof(half));

    // convert x and y from float type to half type
    dim3 dimBlock(32, 8);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    float2half_kernel<<< dimGrid, dimBlock >>>(d_half_x, X.d_ptr_);
    float2half_kernel<<< dimGrid, dimBlock >>>(d_half_y, Y.d_ptr_);
    
    // start initial 1 operation as a warm start
    hfma_kernel<<< dimGrid, dimBlock >>>(d_half_x, d_half_y, Z.d_ptr_, height, width);

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (int i = 0; i < n_iteration; i++) {
        hfma_kernel<<< dimGrid, dimBlock >>>(d_half_x, d_half_y, Z.d_ptr_, height, width);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    float elapsedTimeMs = sdkGetTimerValue(&timer) / (float)n_iteration;
    float throughput = 3 * 2.f * size / elapsedTimeMs;
    printf("FMA, Throughput = %.3f GFlops, Operation Time= %.3f msec\n", throughput * 1e-6, elapsedTimeMs);

    // cleanup
    sdkDeleteTimer(&timer);
    
    return 0;
}