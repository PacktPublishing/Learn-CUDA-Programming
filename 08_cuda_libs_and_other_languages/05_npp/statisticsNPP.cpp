#include <iostream>
#include <cuda_runtime.h>
#include <npp.h>

void GetData(float** buffer, size_t size)
{
    (*buffer) = (float*) new float[size];

    for (int i = 0; i < size; i++) {
        (*buffer)[i] = float(rand() % 0xFFFF) / RAND_MAX;
    }
}

int GetWorkspaceSize(int signalSize)
{
    int bufferSize, tempBufferSize;

    nppsSumGetBufferSize_32f(signalSize, &tempBufferSize);
    bufferSize = std::max(bufferSize, tempBufferSize);
    nppsMinGetBufferSize_32f(signalSize, &tempBufferSize);
    bufferSize = std::max(bufferSize, tempBufferSize);
    nppsMaxGetBufferSize_32f(signalSize, &tempBufferSize);
    bufferSize = std::max(bufferSize, tempBufferSize);
    nppsMeanGetBufferSize_32f(signalSize, &tempBufferSize);
    bufferSize = std::max(bufferSize, tempBufferSize);
    nppsNormDiffL2GetBufferSize_32f(signalSize, &tempBufferSize);
    bufferSize = std::max(bufferSize, tempBufferSize);

    return bufferSize;
}

int main()
{
    float *h_input1, *d_input1;
    float *h_input2, *d_input2;
    Npp8u *d_workspace;
    size_t buf_size = 64, workspace_size = 0;
    float h_output, *d_output;

    srand(2019);
    GetData(&h_input1, buf_size);
    GetData(&h_input2, buf_size);
    workspace_size = GetWorkspaceSize(buf_size);

    // prepare input / output memory space
    cudaMalloc((void **)&d_input1, buf_size * sizeof(float));
    cudaMalloc((void **)&d_input2, buf_size * sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float));
    cudaMalloc((void **)&d_workspace, workspace_size * sizeof(Npp8u));

    cudaMemcpy(d_input1, h_input1, buf_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, buf_size * sizeof(float), cudaMemcpyHostToDevice);

    nppsSum_32f(d_input1, buf_size, d_output, d_workspace);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Sum: " << h_output << std::endl;

    nppsMin_32f(d_input1, buf_size, d_output, d_workspace);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Min: " << h_output << std::endl;

    nppsMax_32f(d_input1, buf_size, d_output, d_workspace);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Max: " << h_output << std::endl;

    nppsMean_32f(d_input1, buf_size, d_output, d_workspace);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Mean: " << h_output << std::endl;

    nppsNormDiff_L2_32f(d_input1, d_input2, buf_size, d_output, d_workspace); 
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "NormDiffL2: " << h_output << std::endl;

    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
    cudaFree(d_workspace);

    delete[] h_input1;
    delete[] h_input2;

    return 0;
}
