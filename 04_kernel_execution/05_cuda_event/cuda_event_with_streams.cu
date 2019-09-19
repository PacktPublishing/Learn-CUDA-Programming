#include <cstdio>
#include <helper_timer.h>

using namespace std;

__global__ void vecAdd_kernel(float *c, const float* a, const float* b);
void init_buffer(float *data, const int size);

class Operator
{
private:
    int index;
    StopWatchInterface *p_timer;

    static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData);
    void print_time();

    cudaEvent_t start, stop;

protected:
    cudaStream_t stream = nullptr;

public:
    Operator(bool create_stream = true) {
        if (create_stream)
            cudaStreamCreate(&stream);
        sdkCreateTimer(&p_timer);

        // create CUDA events
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~Operator() {
        if (stream != nullptr)
            cudaStreamDestroy(stream);
        sdkDeleteTimer(&p_timer);

        // terminate CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void set_index(int idx) { index = idx; }
    void async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize);
    void print_kernel_time();

}; // Operator

void Operator::CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData) {
    Operator* this_ = (Operator*) userData;
    this_->print_time();
}

void Operator::print_time() {
    // end timer
    sdkStopTimer(&p_timer);
    float elapsed_time_msed = sdkGetTimerValue(&p_timer);
    printf("stream %2d - elapsed %.3f ms \n", index, elapsed_time_msed);
}

void Operator::print_kernel_time() {
    float elapsed_time_msed = 0.f;
    cudaEventElapsedTime(&elapsed_time_msed, start, stop);
    printf("kernel in stream %2d - elapsed %.3f ms \n", index, elapsed_time_msed);
}

void Operator::async_operation(float *h_c, const float *h_a, const float *h_b,
                          float *d_c, float *d_a, float *d_b,
                          const int size, const int bufsize)
{
    // start timer
    sdkStartTimer(&p_timer);

    // copy host -> device
    cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

    // record the event before the kernel execution
    cudaEventRecord(start, stream);

    // launch cuda kernel
    dim3 dimBlock(256);
    dim3 dimGrid(size / dimBlock.x);
    vecAdd_kernel<<< dimGrid, dimBlock, 0, stream >>>(d_c, d_a, d_b);

    // record the event right after the kernel execution finished
    cudaEventRecord(stop, stream);

    // copy device -> host
    cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

    // what happen if we include CUDA event synchronize?
    // QUIZ: cudaEventSynchronize(stop);

    // register callback function
    cudaStreamAddCallback(stream, Operator::Callback, this, 0);
}

class Operator_with_priority: public Operator {
public:
    Operator_with_priority() : Operator(false) {}

    void set_priority(int priority) {
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
    }
};

int main(int argc, char* argv[])
{
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    int size = 1 << 24;
    int bufsize = size * sizeof(float);
    int num_operator = 4;

    if (argc != 1)
        num_operator = atoi(argv[1]);

    // check the current device supports CUDA stream's prority
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 
    if (prop.streamPrioritiesSupported == 0) {
        printf("This device does not support priority streams");
        return 1;
    }

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

    // create list of operation elements
    Operator_with_priority *ls_operator = new Operator_with_priority[num_operator];

    // Get Priority range
    int priority_low, priority_high;
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    printf("Priority Range: low(%d), high(%d)\n", priority_low, priority_high);

    // start to measure the execution time
    sdkStartTimer(&timer);
    
    // execute each operator collesponding data
    // priority setting for each CUDA stream
    for (int i = 0; i < num_operator; i++) {
        // int offset = i * size / num_operator;
        ls_operator[i].set_index(i);
        if (i + 1 == num_operator)
            ls_operator[i].set_priority(priority_high);
        else
            ls_operator[i].set_priority(priority_low);
    }

    // operation (copy(H2D), kernel execution, copy(D2H))
    for (int i = 0; i < num_operator; i++) {
        int offset = i * size / num_operator;
        ls_operator[i].async_operation(&h_c[offset], &h_a[offset], &h_b[offset],
                                       &d_c[offset], &d_a[offset], &d_b[offset],
                                       size / num_operator, bufsize / num_operator);
    }

    // synchronize all the stream operation
    cudaDeviceSynchronize();

    // stop to measure the execution time    
    sdkStopTimer(&timer);

    // print each cuda stream execution time
    for (int i = 0; i < num_operator; i++)
        ls_operator[i].print_kernel_time(); 

    // print out the result
    int print_idx = 256;
    printf("compared a sample result...\n");
    printf("host: %.6f, device: %.6f\n",  h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

    // Compute and print the performance
    float elapsed_time_msed = sdkGetTimerValue(&timer);
    float bandwidth = 3 * bufsize * sizeof(float) / elapsed_time_msed / 1e6;
    printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

    // delete timer
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

__global__ void
vecAdd_kernel(float *c, const float* a, const float* b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 500; i++)
        c[idx] = a[idx] + b[idx];
}

void init_buffer(float *data, const int size)
{
    for (int i = 0; i < size; i++) 
        data[i] = rand() / (float)RAND_MAX;
}