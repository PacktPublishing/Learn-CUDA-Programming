#include <cstdio>
#include <helper_timer.h>

using namespace std;

__global__ void vecAdd_kernel(float *c, const float* a, const float* b);
void init_buffer(float *data, const int size);

class Operator
{
private:
    int _index;
    cudaStream_t _stream;
    StopWatchInterface *_p_timer;
    cudaEvent_t _start, _stop;

public:
    Operator() {
        cudaStreamCreate(&_stream);
        sdkCreateTimer(&_p_timer);

        // create cuda event
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
    }

    ~Operator() {
        cudaStreamDestroy(_stream);
        sdkDeleteTimer(&_p_timer);

        // destroy cuda event
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }

    void set_index(int index) { _index = index; }
    void async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize);
    float get_elapsed_time() {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, _start, _stop);
        return milliseconds;
    }
    
}; // Operator

void Operator::async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize)
{
    // start timer
    sdkStartTimer(&_p_timer);

    // copy host -> device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, _stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, _stream);
    
    cudaEventRecord(_start, _stream);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<< dimGrid, dimBlock, 0, _stream >>>(d_c, d_a, d_b);

    cudaEventRecord(_stop, _stream);

    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, _stream);

    // quiz: what would happen if this API call below is activated?
    // cudaEventSynchronize(_stop);
}

float Operator::get_elapsed_time() {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, _start, _stop);
    return milliseconds;
}

int main()
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsize = size * sizeof(float);
    int num_operator = 4;

    // initialize timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);
    
    // allocate host memories
    cudaMallocHost((void**)&h_a, bufsize);
    cudaMallocHost((void**)&h_b, bufsize);
    cudaMallocHost((void**)&h_c, bufsize);

    // initialize host values
    srand(2019);
    init_buffer(h_a, size);
    init_buffer(h_b, size);
    init_buffer(h_c, size);

    // allocate device memories
    cudaMalloc((void**)&d_a, bufsize);
    cudaMalloc((void**)&d_b, bufsize);
    cudaMalloc((void**)&d_c, bufsize);

    sdkStartTimer(&timer);

    // create list of operation elements
    Operator *ls_operator = new Operator[num_operator];
    
    // execute each operator collesponding data
    for (int i = 0; i < num_operator; i++) {
        int offset = i * size / num_operator;
        ls_operator[i].set_index(i);
        ls_operator[i].async_operation(&h_c[offset], &h_a[offset], &h_b[offset],
                                       &d_c[offset], &d_a[offset], &d_b[offset],
                                       size / num_operator, bufsize / num_operator);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);

    printf("kernel execution time\n");
    for (int i = 0; i < num_operator; i++) {
        printf("stream %2d - elapsed time %.3f ms\n", i, ls_operator[i].get_elapsed_time());
    }

    // print out the result
    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %.6f, device: %.6f\n",  h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

    // Compute and print the performance
    float elapsed_time_msed = sdkGetTimerValue(&timer);
    float bandwidth = 3 * bufsize * sizeof(float) / elapsed_time_msed / 1e6;
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    sdkDeleteTimer(&timer);

    // terminate operators
    delete [] ls_operator;

    // terminate device memories
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // terminate host memories
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    
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