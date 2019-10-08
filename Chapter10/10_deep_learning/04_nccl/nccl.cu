#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <string>

#include <cuda_runtime.h>
#include <nccl.h>
#include <cuda_profiler_api.h>

#include <chrono>
#include <thread>
#include <pthread.h>

#define BUSY_WAITING 1

void generate_data(float *ptr, int length);

#define checkNcclErrors(err) { \
    if (err != ncclSuccess)    \
    {                          \
        fprintf(stderr, "checkNcclErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    err, ncclGetErrorString(err), __FILE__, __LINE__);                          \
            exit(-1);                                                                           \
    }                                                                                           \
}

typedef struct device
{
    float *d_send;
    float *d_recv;
    cudaStream_t stream;
} device_t;

int main(int argc, char *argv[]) 
{
    int num_dev = 0;

    // prepare the GPU devices
    cudaGetDeviceCount(&num_dev);

    ncclComm_t *ls_comms = new ncclComm_t[num_dev];
    int *dev_ids = new int[num_dev];
    for (int i = 0; i < num_dev; i++)
        dev_ids[i] = i;

    // prepare data
    unsigned long long size = 512 * 1024 * 1024; // 2 GB

    // allocate device buffers and initialize device handles
    device_t *ls_dev = new device_t[num_dev];
    for (int i = 0; i < num_dev; i++)
    {
        cudaSetDevice(i);
        cudaMalloc((void**)&ls_dev[i].d_send, sizeof(float) * size);
        cudaMalloc((void**)&ls_dev[i].d_recv, sizeof(float) * size);
        cudaMemset(ls_dev[i].d_send, 0, sizeof(float) * size);
        cudaMemset(ls_dev[i].d_recv, 0, sizeof(float) * size);
        cudaStreamCreate(&ls_dev[i].stream);
    }

    // start nvprof profiler
    cudaProfilerStart();

    // initialize nccl communications
    checkNcclErrors(ncclCommInitAll(ls_comms, num_dev, dev_ids));

    // Calling NCCL communcation
    for (unsigned long long test_size = 1024; test_size <= size; test_size <<= 1)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < 1; t++)
        {
            checkNcclErrors(ncclGroupStart());
            for (int i = 0; i < num_dev; i++) {
                checkNcclErrors(ncclAllReduce((const void*)ls_dev[i].d_send, (void*)ls_dev[i].d_recv, test_size, ncclFloat, ncclSum, ls_comms[i], ls_dev[i].stream));
            }
            checkNcclErrors(ncclGroupEnd());

            // synchronize on CUDA stream to wait for the completion of all the communications
#if (BUSY_WAITING)
            for (int i = 0; i < num_dev; i++)
            {
                cudaError_t err = cudaErrorNotReady;
                while (err == cudaErrorNotReady) { 
                    err = cudaStreamQuery(ls_dev[i].stream);
                    pthread_yield();
                }
            }
#else
            for (int i = 0; i < num_dev; i++) {
                cudaSetDevice(i);
                cudaStreamSynchronize(ls_dev[i].stream);
            }
#endif
        }
        cudaProfilerStop();
        auto end = std::chrono::high_resolution_clock::now();

        // report the performance
        // https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
        float elapsed_time_in_ms = std::chrono::duration<float, std::milli>(end - start).count();
        elapsed_time_in_ms /= 1.f;
        float algbw = sizeof(float) * test_size / elapsed_time_in_ms;
        int perf_factor = 2 * (num_dev - 1) / num_dev;
        std::cout << "bandwidth: " << std::fixed << std::setw(6) << std::setprecision(2) << algbw * perf_factor / 1e+6 << " GB/s";
        std::cout << " for " << std::setw(6) << sizeof(float) * test_size / 1024 << " kbytes." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // free device memory
    for (int i = 0; i < num_dev; i++)
    {
        cudaSetDevice(i);
        cudaFree(ls_dev[i].d_send);
        cudaFree(ls_dev[i].d_recv);
        cudaStreamDestroy(ls_dev[i].stream);
    }

    // destroy nccl communicator objects
    for (int i = 0; i < num_dev; i++)
        ncclCommDestroy(ls_comms[i]);

    // destory resource handles
    delete [] ls_comms;
    delete [] dev_ids;
    delete [] ls_dev;
}

// generate input data
void generate_data(float *ptr, int length)
{
    // fill the buffer with random generated unsigned integers
    for (int i = 0; i < length; i++)
        ptr[i] = (rand() - RAND_MAX/2) / (float)RAND_MAX;
}