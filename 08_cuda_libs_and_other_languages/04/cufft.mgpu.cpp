#include <iostream>
#include <cuda_runtime.h>
#include <cufftXt.h>

typedef cufftReal    Real;
typedef cufftComplex Complex;

int main(int argc, char *argv[])
{
    long long sample_size = 1 << 20;   // 1,048,576
    long long batch_size = 1 << 10;    // 512
    int n_gpu = 2, total_gpu = -1;

    cufftHandle cufft_plan;
    Complex *h_sample_input, *h_sample_output;

    // checking number of GPUs
    if (argc == 2)
        n_gpu = atoi(argv[1]);

    if (n_gpu == 1) {
        std::cout << "Required 2 GPUs at least" << std::endl;
        return -1;
    }

    cudaGetDeviceCount(&total_gpu);

    if (n_gpu > total_gpu) {
        std::cout << "Request too many GPUs.. Limiting " << total_gpu << std::endl;
        n_gpu = total_gpu;
    }
    std::cout << "Execute with " << n_gpu << " in " << total_gpu << " GPUs" << std::endl;

    // create cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // create input data
    h_sample_input = (Complex*) new Complex[sample_size * batch_size];
    h_sample_output = (Complex*) new Complex[sample_size * batch_size];

    // 1. Create Empty plan
    cufftCreate(&cufft_plan);

    // 2. Set multiple gpu
    int *devices = (int*) new int[n_gpu];
    for (int i = 0; i < n_gpu; i++)
        devices[i] = i;
    cufftXtSetGPUs(cufft_plan, n_gpu, devices);

    // 3. create the plan
    size_t *workSize = (size_t*) new size_t[n_gpu];
    cufftXtMakePlanMany(cufft_plan, 1, &sample_size,
                        nullptr, 1, 1, CUDA_C_32F,
                        nullptr, 1, 1, CUDA_C_32F,
                        batch_size, workSize, CUDA_C_32F);

    // 4. Malloc data on multiple gpus
    cudaLibXtDesc *d_sample, *d_sample_output;
    cufftXtMalloc(cufft_plan, &d_sample, CUFFT_XT_FORMAT_INPLACE);
    cufftXtMalloc(cufft_plan, &d_sample_output, CUFFT_XT_FORMAT_INPLACE);

    // 5. Copy data from host to multiple GPUs
    cufftXtMemcpy(cufft_plan, d_sample, h_sample_input, CUFFT_COPY_HOST_TO_DEVICE);

    cudaEventRecord(start);     // record start event

    // 6. Execute FFT on multiple GPUs
    cufftXtExecDescriptor(cufft_plan, d_sample, d_sample, CUFFT_FORWARD);

    cudaEventRecord(stop);      // record stop event

    // 7. Execute ifft on multiple GPUs
    cufftXtExecDescriptor(cufft_plan, d_sample_output, d_sample, CUFFT_INVERSE);

    // 8. Copy the result to the host
    cufftXtMemcpy(cufft_plan, h_sample_output, d_sample, CUFFT_COPY_DEVICE_TO_HOST);
    
    // elapsed time estimation
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed in " << elapsedTime << " ms using " << n_gpu << " GPUs." << std::endl;


    delete[] h_sample_input;
    delete[] h_sample_output;
    delete[] workSize;
    delete[] devices;

    cufftXtFree(d_sample);
    cufftXtFree(d_sample_output);
    cufftDestroy(cufft_plan);
    
    return 0;
}
