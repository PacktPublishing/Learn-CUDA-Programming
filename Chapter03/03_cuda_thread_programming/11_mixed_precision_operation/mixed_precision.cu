#include "util.cuh"
#include <cuda_fp16.h>
#include <helper_timer.h>
#include <cooperative_groups.h>
#include <cstdio>

using namespace cooperative_groups;

// FMA numerical arithmetic function in GPU @FP32
// y = x * y + z
// in this kernel, assuming we have transposed matrix y
__global__ void fmaf_kernel(float *d_x, float *d_y, float *d_z, int size)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx_x; i < size; i += stride) {
        d_z[i] = fmaf(d_x[i], d_y[i], 0.f);
    }
}

void fmaf_host(float *h_x, float *h_y, float *h_z, int size)
{
    #pragma omp parallel
    {
    #pragma omp for
        for (int i = 0; i < size; i++)
            h_z[i] = h_x[i] * h_y[i] + 0.f;
    }
}

int main()
{
    CBuffer<float> X, Y, Z;
    int size = 1 << 26;

    srand(2019);

    // initialize host buffers
    X.init(size, true);
    Y.init(size, true);
    Z.init(size, true);

    // initalize gpu buffers
    X.cuda();
    Y.cuda();
    Z.cuda();

    // getting number of blocks for stride-loop
    int n_threads = 256;
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, fmaf_kernel, n_threads, n_threads*sizeof(float2));
    int n_blocks = min(num_blocks_per_sm * num_sms, (size/2 + n_threads - 1) / n_threads);

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    fmaf_kernel<<< n_blocks, n_threads, n_threads * sizeof(float) >>>(X.d_ptr_, Y.d_ptr_, Z.d_ptr_, size);
    
    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    float elapsedTimeMs = sdkGetTimerValue(&timer);
    float ops = size / elapsedTimeMs * 1e-6;
    printf("FMA, FLOPS = %.3f GFlops, Operation Time= %.3f msec\n", ops, elapsedTimeMs);

    fmaf_host(X.h_ptr_, Y.h_ptr_, Z.h_ptr_, size);

    int diff_count = Z.diff_count();
    (diff_count == 0) ? printf("Success!!\n") : printf("Counted diff!! (%d times)\n", diff_count);

    // cleanup
    sdkDeleteTimer(&timer);
    
    return 0;
}
