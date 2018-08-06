
/**
// CUDA also supports 8-bit and 16-bit dot product in the header sm_61_intrinsics.h.
// SM_61, GP102, GP104, GP106


__device__ int __dp4a(int srcA, int srcB, int c);
__device__ int __dp4a(char4 srcA, char4 srcB, int c);
__device__ unsigned int __dp4a(unsigned int srcA, unsigned int srcB, unsigned int c);
__device__ unsigned int __dp4a(uchar4 srcA, uchar4 srcB, unsigned int c);

// for convenience, there are both int and char4 versions of DP4A intrinsics in both signed and unsigned.
// Both version assums that the four vector elements of A and B are packed into the corresponding bytes of a single 32-bit word.
// whird ```char4``` and ```uchar``` types shows CUDA's struct type to express the bytes implicitly,
// the packed resualt is represented by 32-bit sized version.

// On the other hand, DP2A has a 'high' and 'low' version for selecting either high or low two corresponding input bytes.

__device__ int __dp2a_lo(int srcA, int srcB, int c);
__device__ unsigned int __dp2a_lo(unsigned int srcA, unsigned int srcB, unsigned int c);

// Vector-style [_lo]
__device__ int __dp2a_lo(short2 srcA, char4 srcB, int c);
__device__ unsigned int __dp2a_lo(ushort2 srcA, uchar4 srcB, unsigned int c);

// Generic [_hi]
__device__ int __dp2a_hi(int srcA, int srcB, int c);
__device__ unsigned int __dp2a_hi(unsigned int srcA, unsigned int srcB, unsigned int c);

// Vector-style [_hi]
__device__ int __dp2a_hi(short2 srcA, char4 srcB, int c);
__device__ unsigned int __dp2a_hi(ushort2 srcA, uchar4 srcB, unsigned int c);

// Keep in mind that DP2A and DP4A are available on Tesla, GeForce, and Quadro.
*/

#include <sm_61_intrinsics.h>
#include <helper_timer.h>
#include <helper_math.h>
#include <cooperative_groups.h>
#include "util.cuh"
#include <cstdio>

using namespace cooperative_groups;

__global__ void dp4a_kernel(char *d_x, char *d_y, int *d_z, int height, int width)
{
    grid_group grid = this_grid();
    thread_block block = this_thread_block();

    int idx_x = block.group_dim().x * block.group_index().x + block.thread_index().x;
    int idx_y = block.group_dim().y * block.group_index().y + block.thread_index().y;
    int stride = grid.size();

#if __CUDA_ARCH__ >= 610
    int quater_size = width / 4;
    char4 *quad_x = (char4 *)d_x;
    char4 *quad_y = (char4 *)d_y;
    
    int sum = 0;
    for (int i = 0; i < quater_size; i++) {
        sum += __dp4a(quad_y[idx_y * quater_size + i], quad_x[idx_y * quater_size + i], 0);
    }
    d_z[idx_y * width + idx_x] = sum;
#else
    int sum = 0;
    for (int i = 0; i < width; i++ ) {
        sum += d_y[idx_y * width + i] * d_x[idx_y * width + i];
    }
#endif
}

int main()
{
    CBuffer<char> X, Y;
    CBuffer<int> Z;
    int height = 1 << 12;
    int width = 1 << 12;
    int size = height * width;
    int n_iteration = 100;

    srand(2019);

    // initialize host buffers
    X.init(size, true);
    Y.init(size, true);
    Z.init(size);

    // initalize gpu buffers
    X.cuda(true);
    Y.cuda(true);
    Z.cuda(false);

    // convert x and y from float type to half type
    dim3 dimBlock(64, 8);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
        
    // start initial 1 operation as a warm start
    dp4a_kernel<<< dimGrid, dimBlock >>>(X.d_ptr_, Y.d_ptr_, Z.d_ptr_, height, width);

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (int i = 0; i < n_iteration; i++) {
        dp4a_kernel<<< dimGrid, dimBlock >>>(X.d_ptr_, Y.d_ptr_, Z.d_ptr_, height, width);
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
