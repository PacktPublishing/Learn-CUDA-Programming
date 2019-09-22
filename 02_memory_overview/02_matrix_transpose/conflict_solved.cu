#include<stdio.h>
#include<stdlib.h>

#define N 1024
#define BLOCK_SIZE 32 

__global__ void matrix_transpose_naive(int *input, int *output) {

	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;
	int index = indexY * N + indexX;
	int transposedIndex = indexX * N + indexY;

        output[transposedIndex] = input[index];
}

__global__ void matrix_transpose_shared(int *input, int *output) {

	__shared__ int sharedMemory [BLOCK_SIZE] [BLOCK_SIZE + 1];

	//global index	
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;

	//transposed index	
	int tindexX = threadIdx.x + blockIdx.y * blockDim.x;
	int tindexY = threadIdx.y + blockIdx.x * blockDim.y;

	int localIndexX = threadIdx.x;
	int localIndexY = threadIdx.y;

	int index = indexY * N + indexX;
	int transposedIndex = tindexY * N + tindexX;
	//reading from global memory in coalesed manner and performing tanspose in shared memory
	sharedMemory[localIndexX][localIndexY] = input[index];

	__syncthreads();
	//writing into global memory in coalesed fashion via transposed data in shared memory
	output[transposedIndex] = sharedMemory[localIndexY][localIndexX];
}

//basically just fills the array with index.
void fill_array(int *data) {
	for(int idx=0;idx<(N*N);idx++)
		data[idx] = idx;
}

void print_output(int *a, int *b) {
	printf("\n Original Matrix::\n");
	for(int idx=0;idx<(N*N);idx++) {
		if(idx%N == 0)
			printf("\n");
		printf(" %d ",  a[idx]);
	}
	printf("\n Transposed Matrix::\n");
	for(int idx=0;idx<(N*N);idx++) {
		if(idx%N == 0)
			printf("\n");
		printf(" %d ",  b[idx]);
	}
}
int main(void) {
	int *a, *b;
        int *d_a, *d_b; // device copies of a, b, c

	int size = N * N *sizeof(int);

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); fill_array(a);
	b = (int *)malloc(size);

        // Alloc space for device copies of a, b, c
        cudaMalloc((void **)&d_a, size);
        cudaMalloc((void **)&d_b, size);

       // Copy inputs to device
        cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE,1);
	dim3 gridSize(N/BLOCK_SIZE,N/BLOCK_SIZE,1);

	matrix_transpose_naive<<<gridSize,blockSize>>>(d_a,d_b);

        // Copy result back to host
        cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

	//print_output(a,b);


	matrix_transpose_shared<<<gridSize,blockSize>>>(d_a,d_b);

        // Copy result back to host
        cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

	//print_output(a,b);



	free(a); free(b);
        cudaFree(d_a); cudaFree(d_b); 

	return 0;
}
