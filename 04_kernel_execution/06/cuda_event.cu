#include <cstdio>

using namespace std;

__global__ void vecAdd_kernel(float *c, const float* a, const float* b);

void cuda_async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize, cudaStream_t stream = 0)
{
    // create cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // transfer data from host to device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

    cudaEventRecord(start, stream);

    // launch kernel
    dim3 block_size(256);
    dim3 grid_size(size / block_size.x);
    vecAdd_kernel<<< grid_size, block_size, 0, stream >>>(d_c, d_a, d_b);

    cudaEventRecord(stop, stream);

    // transfer the result
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

    // synchronization with the event
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f ms\n", milliseconds);

    // delete cuda event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void init_buffer(float *data, const int size);

int main()
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 16;
    int bufsize = size * sizeof(float);
    cudaStream_t stream;

    // allocate host memories
    h_a = new float[size];
    h_b = new float[size];
    h_c = new float[size];

    // initialize host values
    srand(2019);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    // create cuda stream
    cudaStreamCreate(&stream);

    // allocate device memories
    cudaMalloc((void**)&d_a, bufsize);
    cudaMalloc((void**)&d_b, bufsize);
    cudaMalloc((void**)&d_c, bufsize);
    
    cuda_async_operation(h_c, h_a, h_b, d_c, d_a, d_b, size, bufsize, stream);

    // print out the result
    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %.6f, device: %.6f\n",  h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

    // terminate cuda stream
    cudaStreamDestroy(stream);

    // terminate device memories
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // terminate host memories
    delete [] h_a;
    delete [] h_b;
    delete [] h_c;
    
    return 0;
}

void init_buffer(float *data, const int size)
{
    for (int i = 0; i < size; i++) 
        data[i] = rand() / (float)RAND_MAX;
}

__global__ void
vecAdd_kernel(float *c, const float* a, const float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
       c[idx] = a[idx] + b[idx];
}