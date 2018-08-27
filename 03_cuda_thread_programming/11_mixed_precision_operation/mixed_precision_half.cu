
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
__global__ void hfma_kernel(half *d_x, half *d_y, float *d_z, int size)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    half2 *dual_x = reinterpret_cast<half2*>(d_x);
    half2 *dual_y = reinterpret_cast<half2*>(d_y);
    float2 *dual_z = reinterpret_cast<float2*>(d_z);

    extern __shared__ float2 s_data[];

#if __CUDA_ARCH__ >= 530
    for (int i = idx_x; i < size; i+=stride) {
        dual_z[i] = __half22float2(__hmul2(dual_y[i], dual_x[i]));
    }
#else
    for (int i = idx_x; i < size; i+=stride) {
        dual_z[i] = __half22float2(dual_x[i]) * __half22float2(dual_y[i]);
    }
#endif
}

void fhma_host(half *h_x, half *h_y, float *h_z, int size)
{
    #pragma omp parallel
    {
    #pragma omp for
        for (int i = 0; i < size; i++)
            h_z[i] = __half2float(h_y[i]) * __half2float(h_x[i]);
    }
}

int main()
{
    CBuffer<half> X, Y;
    CBuffer<float> Z;
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
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, hfma_kernel, n_threads, n_threads*sizeof(float2));
    int n_blocks = min(num_blocks_per_sm * num_sms, (size/2 + n_threads - 1) / n_threads);

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    hfma_kernel<<< n_blocks, n_threads, n_threads * sizeof(float2) >>>(X.d_ptr_, Y.d_ptr_, Z.d_ptr_, size/2);

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    float elapsedTimeMs = sdkGetTimerValue(&timer);
    double ops = (float)size / elapsedTimeMs * 1e-6;
    printf("FMA, FLOPS = %.3f GFlops, Operation Time= %.3f msec\n", ops, elapsedTimeMs);

    fhma_host(X.h_ptr_, Y.h_ptr_, Z.h_ptr_, size);

    int diff_count = Z.diff_count();
    (diff_count == 0) ? printf("Success!!\n") : printf("Counted diff!! (%d times)\n", diff_count);

    // cleanup
    sdkDeleteTimer(&timer);
    
    return 0;
}