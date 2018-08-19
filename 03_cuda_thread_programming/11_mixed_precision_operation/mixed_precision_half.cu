
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

// FMA numerical arithmetic function in GPU @FP16
// y = x * y + z
// in this kernel, assuming we have transposed matrix y
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
#else
    float sum = 0.f;
    for (int i = 0; i < width; i++)
        sum += __half2float(d_x[idx_y * width + i]) * __half2float(d_y[idx_y * width + i]);
#endif
    d_z[idx_y * width + idx_x] = sum;
}

void matmul_host(half *h_x, half *h_y, float *h_z, int height, int width)
{
    #pragma omp parallel
    for (int idx_y = 0; idx_y < height; idx_y++) {
        for (int idx_x = 0; idx_x < width; idx_x++) {
            float sum = 0.f;
            for (int i = 0; i < width; i++)
                sum += __half2float(h_y[idx_y * width + i]) * __half2float(h_x[idx_y * width + i]);

            h_z[idx_y * width + idx_x] = sum;
        }
    }
}

int main()
{
    CBuffer<half> X, Y;
    CBuffer<float> Z;
    int height = 1 << 10;
    int width = 1 << 10;
    int size = height * width;
    int n_iteration = 100;

    srand(2019);

    // initialize host buffers
    X.init(size, true);
    Y.init(size, true);
    Z.init(size, false);

    // initalize gpu buffers
    X.cuda();
    Y.cuda();
    Z.cuda();

    // convert x and y from float type to half type
    dim3 dimBlock(32, 8);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    // start initial 1 operation as a warm start
    hfma_kernel<<< dimGrid, dimBlock >>>(X.d_ptr_, Y.d_ptr_, Z.d_ptr_, height, width);

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (int i = 0; i < n_iteration; i++) {
        hfma_kernel<<< dimGrid, dimBlock >>>(X.d_ptr_, Y.d_ptr_, Z.d_ptr_, height, width);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    float elapsedTimeMs = sdkGetTimerValue(&timer) / (float)n_iteration;
    float throughput = 3 * 2.f * size / elapsedTimeMs;
    printf("FMA, Throughput = %.3f GFlops, Operation Time= %.3f msec\n", throughput * 1e-6, elapsedTimeMs);

    matmul_host(X.h_ptr_, Y.h_ptr_, Z.h_ptr_, height, width);

    int diff_count = Z.diff_count();
    (diff_count == 0) ? printf("Success!!\n") : printf("Counted diff!! (%d times)\n", diff_count);

    // cleanup
    sdkDeleteTimer(&timer);
    
    return 0;
}