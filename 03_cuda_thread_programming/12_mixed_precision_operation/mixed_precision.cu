#include "util.cuh"
#include <cuda_fp16.h>
#include <helper_timer.h>
#include <cooperative_groups.h>
#include <cstdio>

using namespace cooperative_groups;

// FMA numerical arithmetic function in GPU @FP32
// y = x * y + z
__global__ void matmul_kernel(float *d_x, float *d_y, float *d_z, int height, int width)
{
    thread_block block = this_thread_block();
    int idx_x = block.group_dim().x * block.group_index().x + block.thread_index().x;
    int idx_y = block.group_dim().y * block.group_index().y + block.thread_index().y;
    
    float sum = 0.f;
    for (int i = 0; i < width; i++)
        sum += d_y[i * width + idx_x] * d_x[idx_y * width + i];

    d_x[idx_y * width + idx_x] = sum;
}

int main()
{
    CBuffer<float> W, X, Y, Z;
    int width = 1 << 12;
    int height = 1 << 12;
    int size = width * height;
    int n_iteration = 100;

    srand(2019);

    // initialize host buffers
    W.init(size);
    X.init(size, true);
    Y.init(size, true);
    Z.init(size, true);

    // initalize gpu buffers
    W.cuda(false);
    X.cuda(true);
    Y.cuda(true);
    Z.cuda(true);

    // start initial 1 operation as a warm start
    dim3 dimBlock(16, 8);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    matmul_kernel<<< dimGrid, dimBlock >>>(X.d_ptr_, Y.d_ptr_, Z.d_ptr_, height, width);

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (int i = 0; i < n_iteration; i++) {
        matmul_kernel<<< dimGrid, dimBlock >>>(X.d_ptr_, Y.d_ptr_, Z.d_ptr_, height, width);
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




