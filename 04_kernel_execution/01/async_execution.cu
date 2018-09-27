#include <cstdio>
#include <cstdlib>

using namespace std;

__global__ void
vecAdd_kernel(float *c, const float* a, const float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];

    if (idx == 0)
        printf("%s\n", __func__);
}

void cuda_sync_operation(float *h_c, const float *h_a, const float *h_b,
                         float *d_c, float *d_a, float *d_b,
                         const int size, const int bufsize)
{
    printf("######## %s ########\n", __func__);
    printf("copy host -> device\n");

    // transfer data from host to device
    cudaMemcpy(d_a, h_a, bufsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bufsize, cudaMemcpyHostToDevice);

    printf("kernel execution\n");

    // launch kernel
    dim3 block_size(256);
    dim3 grid_size(size / block_size.x);
    vecAdd_kernel<<< grid_size, block_size >>>(d_c, d_a, d_b);

    printf("copy device -> host\n");

    // transfer the result
    cudaMemcpy(h_c, d_c, bufsize, cudaMemcpyDeviceToHost);

    printf("device sync\n");

    cudaDeviceSynchronize();

    printf("%s.. done\n", __func__);
}

void cuda_async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize)
{
    printf("######## %s ########\n", __func__);
    printf("copy host -> device\n");

    // transfer data from host to device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice);

    printf("kernel execution\n");

    // launch kernel
    dim3 block_size(256);
    dim3 grid_size(size / block_size.x);
    vecAdd_kernel<<< grid_size, block_size >>>(d_c, d_a, d_b);

    printf("copy device -> host\n");

    // transfer the result
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost);

    printf("device sync\n");

    cudaDeviceSynchronize();

    printf("%s.. done\n", __func__);
}

void init_buffer(float *data, const int size);

int main()
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 16;
    int bufsize = size * sizeof(float);

    // allocate host memories
    h_a = new float[size];
    h_b = new float[size];
    h_c = new float[size];

    // initialize host values
    srand(2019);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    // allocate device memories
    cudaMalloc((void**)&d_a, bufsize);
    cudaMalloc((void**)&d_b, bufsize);
    cudaMalloc((void**)&d_c, bufsize);

    cuda_sync_operation(h_c, h_a, h_b, d_c, d_a, d_b, size, bufsize);
    cuda_async_operation(h_c, h_a, h_b, d_c, d_a, d_b, size, bufsize);

    // print out the result
    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %.6f, device: %.6f\n",  h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

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