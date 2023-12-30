#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h> // Needed or __global__ == unrecognised.
#include "device_launch_parameters.h" // Variable identifiers.
#include <memory>
#include <array>
#include "CudaContainer.h"

const int SIZE = 256;
const int THREADS_PER_BLOCK = 4;
const int NO_OF_BLOCKS = SIZE / THREADS_PER_BLOCK;

__global__ void device_add(int* a, int* b, int* c);
void fill_array(const std::shared_ptr<std::array<int, SIZE>>& out);

int main()
{
	std::cout << "Hello" << std::endl;

	// Host memory allocation.
	std::shared_ptr<std::array<int, SIZE>> a = std::make_shared<std::array<int, SIZE>>();
	std::shared_ptr<std::array<int, SIZE>> b = std::make_shared<std::array<int, SIZE>>();
	std::shared_ptr<std::array<int, SIZE>> c = std::make_shared<std::array<int, SIZE>>();
	// Device memory allocation.
	std::shared_ptr<CudaContainer<int>> d_a = std::make_shared<CudaContainer<int>>(SIZE);
	std::shared_ptr<CudaContainer<int>> d_b = std::make_shared<CudaContainer<int>>(SIZE);
	std::shared_ptr<CudaContainer<int>> d_c = std::make_shared<CudaContainer<int>>(SIZE);

	fill_array(a);
	fill_array(b);

	cudaMemcpy(d_a->data, a.get(), SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b->data, b.get(), SIZE * sizeof(int), cudaMemcpyHostToDevice);

	device_add << <NO_OF_BLOCKS, THREADS_PER_BLOCK >> > (d_a->data, d_b->data, d_c->data);

	cudaDeviceSynchronize();

	cudaMemcpy(c.get(), d_c->data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	// No need to manually call free or cudaFree.
	// Since cudaFree is in the destructor of ~CudaContainer.
	// Which is wrapped in a shared_ptr.

	for (int i = 0; i < SIZE; i++)
		std::cout << (*c)[i] << std::endl;

	return 0;
}

template <class T>
CudaContainer<T>::CudaContainer(int size)
{
	this->size = size;
	cudaMalloc(&data, size * sizeof(T));
}

template <class T>
CudaContainer<T>::~CudaContainer()
{
	cudaFree(data);
}

__global__ void device_add(int* a, int* b, int* c)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Setting index %d to %d + %d\n", index, a[index], b[index]);
	c[index] = a[index] + b[index];
}

void fill_array(const std::shared_ptr<std::array<int, SIZE>>& out)
{
	for (int i = 0; i < SIZE; i++)
		(*out)[i] = i;
}