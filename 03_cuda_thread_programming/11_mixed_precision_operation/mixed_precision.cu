#include "util.cuh"
#include <cuda_fp16.h>
#include <helper_timer.h>
#include <cooperative_groups.h>
#include <cstdio>

using namespace cooperative_groups;

// FMA numerical arithmetic function in GPU @FP32
// y = x * y + z
// in this kernel, assuming we have transposed matrix y
__global__ void matmul_kernel(float *d_x, float *d_y, float *d_z, int height, int width)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    float sum = 0.f;
    for (int i = 0; i < width; i++)
        sum += d_y[idx_y * width + i] * d_x[idx_y * width + i];

    d_z[idx_y * width + idx_x] = sum;
}

void matmul_host(float *h_x, float *h_y, float *h_z, int height, int width)
{
    #pragma omp parallel
    for (int idx_y = 0; idx_y < height; idx_y++) {
        for (int idx_x = 0; idx_x < width; idx_x++) {
            float sum = 0.f;
            for (int i = 0; i < width; i++)
                sum += h_y[idx_y * width + i] * h_x[idx_y * width + i];

            h_z[idx_y * width + idx_x] = sum;
        }
    }
}

int main()
{
    CBuffer<float> W, X, Y, Z;
    int width = 1 << 10;
    int height = 1 << 10;
    int size = width * height;
    int n_iteration = 100;

    srand(2019);

    // initialize host buffers
    X.init(size, true);
    Y.init(size, true);
    Z.init(size, true);

    // initalize gpu buffers
    X.cuda();
    Y.cuda();
    Z.cuda();

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
    float throughput = 2.f * size / elapsedTimeMs;
    printf("FMA, Throughput = %.3f GFlops, Operation Time= %.3f msec\n", throughput * 1e-6, elapsedTimeMs);

    matmul_host(X.h_ptr_, Y.h_ptr_, Z.h_ptr_, height, width);

    int diff_count = Z.diff_count();
    (diff_count == 0) ? printf("Success!!\n") : printf("Counted diff!! (%d times)\n", diff_count);

    // cleanup
    sdkDeleteTimer(&timer);
    
    return 0;
}




