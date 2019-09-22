#include <iostream>
#include<stdio.h>
#include <math.h>

#define STRIDE_64K 65536

__global__ void init(int n, float *x, float *y) {

  int lane_id = threadIdx.x & 31;
  size_t warp_id = (threadIdx.x + blockIdx.x * blockDim.x) >> 5;
  size_t warps_per_grid = (blockDim.x * gridDim.x) >> 5;
  size_t warp_total = ((sizeof(float)*n) + STRIDE_64K-1) / STRIDE_64K;


if(blockIdx.x==0 && threadIdx.x==0) {
	//printf("\n TId[%d] ", threadIdx.x);
	//printf(" WId[%u] ", warp_id);
	//printf(" LId[%u] ", lane_id);
	//printf(" WperG[%u] ", warps_per_grid);
	//printf(" wTot[%u] ", warp_total);
	//printf(" rep[%d] ", STRIDE_64K/sizeof(float)/32);
}
  for(; warp_id < warp_total; warp_id += warps_per_grid) {
    #pragma unroll
    for(int rep = 0; rep < STRIDE_64K/sizeof(float)/32; rep++) {
      size_t ind = warp_id * STRIDE_64K/sizeof(float) + rep * 32 + lane_id;
      if (ind < n) {
        x[ind] = 1.0f;
//if(blockIdx.x==0 && threadIdx.x==0) {
//	printf(" \nind[%d] ", ind);
//} 
        y[ind] = 2.0f;
      }
    }
  }

}
 
// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
 
int main(void)
{
  int N = 1<<20;
  float *x, *y;
 
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));
 
 
  // Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  size_t warp_total = ((sizeof(float)*N) + STRIDE_64K-1) / STRIDE_64K;
  int numBlocksInit = (warp_total*32) / blockSize;
  
  init<<<numBlocksInit, blockSize>>>(N, x, y);
  add<<<numBlocks, blockSize>>>(N, x, y);
 
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
 
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
 
  // Free memory
  cudaFree(x);
  cudaFree(y);
 
  return 0;
}
