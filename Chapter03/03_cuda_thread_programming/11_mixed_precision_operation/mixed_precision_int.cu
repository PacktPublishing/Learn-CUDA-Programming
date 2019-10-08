
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

// FMA numerical arithmetic function in GPU @INT8
// y = x * y + z
// in this kernel, assuming we have transposed matrix y
__global__ void dp4a_kernel(char *d_x, char *d_y, int *d_z, int size)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

#if __CUDA_ARCH__ >= 610
    char4 *quad_x = (char4 *)d_x;
    char4 *quad_y = (char4 *)d_y;
    
    for (int i = idx_x; i < size; i+=stride)
        d_z[i] = __dp4a(quad_y[i], quad_x[i], 0);
#else
    for (int i = idx_x; i < size; i+=4*stride) {
        int sum = 0;
        for (int j = 0; j < 4; j++)
            sum += d_y[4 * i + j] * d_x[4 * i + j];
        d_z[i] = sum + 0; 
    }
#endif
}

void dp4a_host(char *h_x, char *h_y, int *h_z, int size)
{
    #pragma omp parallel
    {
    #pragma omp for
        for (int i = 0; i < size; i++) {
            int sum = 0;
            for (int j = 0; j < 4; j++) 
                sum += (int)h_y[4 * i + j] * (int)h_x[4 * i + j];
            h_z[i] = sum; 
        }
    }
}

int main()
{
    CBuffer<char> X, Y;
    CBuffer<int> Z;
    int size = 1 << 26;

    srand(2019);

    // initialize host buffers
    X.init(size, true);
    Y.init(size, true);
    Z.init(size/4, true);

    // initalize gpu buffers
    X.cuda();
    Y.cuda();
    Z.cuda();

    // getting number of blocks for stride-loop
    int n_threads = 256;
    int num_sms;
    int num_blocks_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, dp4a_kernel, n_threads, n_threads*sizeof(int));
    int n_blocks = min(num_blocks_per_sm * num_sms, (size/4 + n_threads - 1) / n_threads);
        
    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    dp4a_kernel<<< n_blocks, n_threads, n_threads * sizeof(int) >>>(X.d_ptr_, Y.d_ptr_, Z.d_ptr_, size/4);

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    float elapsedTimeMs = sdkGetTimerValue(&timer);
    float ops = size / elapsedTimeMs * 1e-6;
    printf("IMA, OPS = %.3f Gops, Operation Time= %.3f msec\n", ops, elapsedTimeMs);

    dp4a_host(X.h_ptr_, Y.h_ptr_, Z.h_ptr_, size/4);

    int diff_count = Z.diff_count();
    (diff_count == 0) ? printf("Success!!\n") : printf("Counted diff!! (%d times)\n", diff_count);

    // cleanup
    sdkDeleteTimer(&timer);

    return 0;
}
