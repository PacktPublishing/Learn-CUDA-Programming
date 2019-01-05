#include <iostream>
#include <cuda_runtime.h>
#include <npp.h>

void GetSum(float *d_buffer, size_t size, Npp32f *d_output)
{
    int hpBufferSize;
    Npp8u *pDeviceBuffer;

    nppsSumGetBufferSize_32f(size, &hpBufferSize);
    cudaMalloc((void **)&pDeviceBuffer, hpBufferSize);
    nppsSum_32f(d_buffer, size, d_output, pDeviceBuffer);
    cudaFree(pDeviceBuffer);
}

void GetMin(float *d_buffer, size_t size, Npp32f *d_output)
{
    int hpBufferSize;
    Npp8u *pDeviceBuffer;

    nppsMinGetBufferSize_32f(size, &hpBufferSize);
    cudaMalloc((void **)&pDeviceBuffer, hpBufferSize);
    nppsMin_32f(d_buffer, size, d_output, pDeviceBuffer);
    cudaFree(pDeviceBuffer);
}

void GetMax(float *d_buffer, size_t size, Npp32f *d_output)
{
    int hpBufferSize;
    Npp8u *pDeviceBuffer;

    nppsMaxGetBufferSize_32f(size, &hpBufferSize);
    cudaMalloc((void **)&pDeviceBuffer, hpBufferSize);
    nppsMax_32f(d_buffer, size, d_output, pDeviceBuffer);
    cudaFree(pDeviceBuffer);
}

void GetMean(float *d_buffer, size_t size, Npp32f *d_output)
{
    int hpBufferSize;
    Npp8u *pDeviceBuffer;
    
    nppsMeanGetBufferSize_32f(size, &hpBufferSize);
    cudaMalloc((void **)&pDeviceBuffer, hpBufferSize);
    nppsMean_32f(d_buffer, size, d_output, pDeviceBuffer);
    cudaFree(pDeviceBuffer);
}

void GetNormDiffL2(float *d_buffer1, float *d_buffer2, size_t size, Npp32f *d_output)
{
    int hpBufferSize;
    Npp8u *pDeviceBuffer;

    nppsNormDiffL2GetBufferSize_32f(size, &hpBufferSize);
    cudaMalloc((void **)&pDeviceBuffer, hpBufferSize);
    nppsNormDiff_L2_32f(d_buffer1, d_buffer2, size, d_output, pDeviceBuffer);    
    cudaFree(pDeviceBuffer);
}

void GetData(float** buffer, size_t size)
{
    (*buffer) = (float*) new float[size];

    for (int i = 0; i < size; i++) {
        (*buffer)[i] = float(rand() % 0xFFFF) / RAND_MAX;
        //(*buffer)[i] = 1;
    }
}

int main()
{
    float *h_buffer, *d_buffer;
    float *h_buffer_tmp, *d_buffer_tmp;
    size_t buf_size = 64;
    float h_output, *d_output;

    srand(2019);
    GetData(&h_buffer, buf_size);

    // prepare input / output memory space
    cudaMalloc((void **)&d_buffer, buf_size * sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float));
    cudaMemcpy(d_buffer, h_buffer, buf_size * sizeof(float), cudaMemcpyHostToDevice);

    GetSum(d_buffer, buf_size, d_output);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Sum: " << h_output << std::endl;

    GetMin(d_buffer, buf_size, d_output);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Min: " << h_output << std::endl;

    GetMax(d_buffer, buf_size, d_output);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Max: " << h_output << std::endl;

    GetMean(d_buffer, buf_size, d_output);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Mean: " << h_output << std::endl;

    h_output = 0.f;
    GetData(&h_buffer_tmp, buf_size);
    cudaMalloc((void **)&d_buffer_tmp, buf_size * sizeof(float));
    cudaMemcpy(d_buffer_tmp, h_buffer_tmp, buf_size * sizeof(float), cudaMemcpyHostToDevice);
    GetNormDiffL2(d_buffer, d_buffer_tmp, buf_size, d_output);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "NormDiffL2: " << h_output << std::endl;

    cudaFree(d_buffer);
    cudaFree(d_buffer_tmp);

    delete[] h_buffer;
    delete[] h_buffer_tmp;

    return 0;
}
