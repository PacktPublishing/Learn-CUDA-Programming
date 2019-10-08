#include <cudnn.h>
#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <cublas_v2.h>

#include "../01_ann/src/helper.h"

void cublas_operation(int num_linear_layers, unsigned long long  num_ops, int input_size, int hidden_size, int seq_length, int batch_size, int num_layers);
void generate_data(const curandGenerator_t generator, float *data, int length);

void rnn_operation(int seq_length, int num_layers, int hidden_size, int input_size, int batch_size, float dropout_rate, bool bidirectional, int mode, int persistent)
{
    // setup inputs and outputs
    // hx, cx, hy, cy, dhy, dcy, dhx, and dcs can be null.
    void *x;                // input
    void *hx  = nullptr;    // input of initial hidden state
    void *cx  = nullptr;    // input of cell state (LSTM)

    void *y;                // output
    void *hy  = nullptr;    // output of final hidden state
    void *cy  = nullptr;    // output of final cell state (LSTM)

    void *dy;               // input of gradient 
    void *dhy = nullptr;    // input of final hidden state
    void *dcy = nullptr;    // input of final cell state (LSTM)

    void *dx;               // output of gradient at the input of rnn
    void *dhx = nullptr;    // output of gradient at the initial hidden state
    void *dcx = nullptr;    // output of gradient at the initial cell state

    // memory allocation
    int input_length  = seq_length * input_size * batch_size;
    int output_length = seq_length * hidden_size * batch_size;
    int hidden_length = hidden_size * batch_size * num_layers;

    if (bidirectional)
    {
        hidden_length *= 2;
        output_length *= 2;
    }

    cudaMalloc((void**)&x,  input_length  * sizeof(float));
    cudaMalloc((void**)&hx, hidden_length * sizeof(float));
    cudaMalloc((void**)&cx, hidden_length * sizeof(float));

    cudaMalloc((void**)&dx, input_length  * sizeof(float));
    cudaMalloc((void**)&dhx,hidden_length * sizeof(float));
    cudaMalloc((void**)&dcx,hidden_length * sizeof(float));

    cudaMalloc((void**)&y,  output_length * sizeof(float));
    cudaMalloc((void**)&hy, hidden_length * sizeof(float));
    cudaMalloc((void**)&cy, hidden_length * sizeof(float));

    cudaMalloc((void**)&dy, output_length * sizeof(float));
    cudaMalloc((void**)&dhy,hidden_length * sizeof(float));
    cudaMalloc((void**)&dcy,hidden_length * sizeof(float));

    // create cudnn handle
    cudnnHandle_t cudnnHandle;
    checkCudnnErrors(cudnnCreate(&cudnnHandle));

    // setup tensor descriptors x/y/dx/dy
    cudnnTensorDescriptor_t x_desc[seq_length], y_desc[seq_length], \
                            dx_desc[seq_length], dy_desc[seq_length];
    cudnnTensorDescriptor_t hx_desc,  cx_desc;
    cudnnTensorDescriptor_t dhx_desc, dcx_desc;
    cudnnTensorDescriptor_t hy_desc,  cy_desc;
    cudnnTensorDescriptor_t dhy_desc, dcy_desc;

    // RNN dimensional information
    int dimA[3];
    int strideA[3];

    // iterate for each element
    for (int i = 0; i < seq_length; i++)
    {
        checkCudnnErrors(cudnnCreateTensorDescriptor(&x_desc[i]));
        checkCudnnErrors(cudnnCreateTensorDescriptor(&y_desc[i]));
        checkCudnnErrors(cudnnCreateTensorDescriptor(&dx_desc[i]));
        checkCudnnErrors(cudnnCreateTensorDescriptor(&dy_desc[i]));

        dimA[0] = batch_size;
        dimA[1] = input_size;
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        checkCudnnErrors(cudnnSetTensorNdDescriptor(x_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
        checkCudnnErrors(cudnnSetTensorNdDescriptor(dx_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));

        dimA[0] = batch_size;
        dimA[1] = bidirectional ? hidden_size * 2 : hidden_size;
        dimA[2] = 1;

        strideA[0] = dimA[2] * dimA[1];
        strideA[1] = dimA[2];
        strideA[2] = 1;

        checkCudnnErrors(cudnnSetTensorNdDescriptor(y_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
        checkCudnnErrors(cudnnSetTensorNdDescriptor(dy_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
    }

    dimA[0] = num_layers * (bidirectional ? 2 : 1);
    dimA[1] = batch_size;
    dimA[2] = hidden_size;

    strideA[0] = dimA[2] * dimA[1];
    strideA[1] = dimA[2];
    strideA[2] = 1;

    checkCudnnErrors(cudnnCreateTensorDescriptor(&hx_desc));
    checkCudnnErrors(cudnnCreateTensorDescriptor(&cx_desc));
    checkCudnnErrors(cudnnCreateTensorDescriptor(&dhx_desc));
    checkCudnnErrors(cudnnCreateTensorDescriptor(&dcx_desc));
    checkCudnnErrors(cudnnCreateTensorDescriptor(&hy_desc));
    checkCudnnErrors(cudnnCreateTensorDescriptor(&cy_desc));
    checkCudnnErrors(cudnnCreateTensorDescriptor(&dhy_desc));
    checkCudnnErrors(cudnnCreateTensorDescriptor(&dcy_desc));

    checkCudnnErrors(cudnnSetTensorNdDescriptor(hx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    checkCudnnErrors(cudnnSetTensorNdDescriptor(cx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    checkCudnnErrors(cudnnSetTensorNdDescriptor(dhx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    checkCudnnErrors(cudnnSetTensorNdDescriptor(dcx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    checkCudnnErrors(cudnnSetTensorNdDescriptor(hy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    checkCudnnErrors(cudnnSetTensorNdDescriptor(cy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    checkCudnnErrors(cudnnSetTensorNdDescriptor(dhy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
    checkCudnnErrors(cudnnSetTensorNdDescriptor(dcy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

    // setup the dropout descriptor
    curandGenerator_t curand_gen;
    unsigned long long seed = 2019UL;
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, seed);

    cudnnDropoutDescriptor_t dropout_desc;
    checkCudnnErrors(cudnnCreateDropoutDescriptor(&dropout_desc));

    size_t state_size;
    void *state;
    checkCudnnErrors(cudnnDropoutGetStatesSize(cudnnHandle, &state_size));
    checkCudaErrors(cudaMalloc(&state, state_size));

    checkCudnnErrors(cudnnSetDropoutDescriptor(dropout_desc, cudnnHandle, dropout_rate, state, state_size, seed));

    /* setup rnn descriptor */
    cudnnRNNDescriptor_t rnn_desc;
    cudnnRNNMode_t rnn_mode;
    cudnnRNNAlgo_t rnn_algo;

    checkCudnnErrors(cudnnCreateRNNDescriptor(&rnn_desc));
    // rnn mode
    switch (mode) {
        case 0: rnn_mode = CUDNN_RNN_RELU;  break;
        case 1: rnn_mode = CUDNN_RNN_TANH;  break;
        case 2: rnn_mode = CUDNN_LSTM;      break;
        case 3: rnn_mode = CUDNN_GRU;       break;
    }

    // rnn algorithm
    switch (persistent) {
        case 0: rnn_algo = CUDNN_RNN_ALGO_STANDARD; break;
        case 1: rnn_algo = CUDNN_RNN_ALGO_PERSIST_STATIC;   break;
        case 2: rnn_algo = CUDNN_RNN_ALGO_PERSIST_DYNAMIC;  break;
    }
    
    checkCudnnErrors(cudnnSetRNNDescriptor_v6(cudnnHandle,
                                        rnn_desc,
                                        hidden_size,
                                        num_layers, dropout_desc, CUDNN_LINEAR_INPUT,
                                        bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
                                        rnn_mode, rnn_algo, CUDNN_DATA_FLOAT));

    // initialize workspaces
    void *weights, *gweights, *workspace, *reserved_space;
    size_t weight_size, workspace_size, reserved_size;
    
    checkCudnnErrors(cudnnGetRNNWorkspaceSize(cudnnHandle, rnn_desc, seq_length, x_desc, &workspace_size));
    checkCudnnErrors(cudnnGetRNNTrainingReserveSize(cudnnHandle, rnn_desc, seq_length, x_desc, &reserved_size));
    checkCudnnErrors(cudnnGetRNNParamsSize(cudnnHandle, rnn_desc, x_desc[0], &weight_size, CUDNN_DATA_FLOAT));
    checkCudaErrors(cudaMalloc((void**)&weights,  weight_size));
    checkCudaErrors(cudaMalloc((void**)&gweights, weight_size));
    cudaMalloc((void**)&workspace, workspace_size);
    cudaMalloc((void**)&reserved_space, reserved_size);

    // initialize filter descriptors
    cudnnFilterDescriptor_t w_desc, dw_desc;
    int dimW[] = { weight_size / sizeof(float), 1, 1 };
    checkCudnnErrors(cudnnCreateFilterDescriptor(&w_desc));
    checkCudnnErrors(cudnnCreateFilterDescriptor(&dw_desc));
    checkCudnnErrors(cudnnSetFilterNdDescriptor(w_desc,  CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));
    checkCudnnErrors(cudnnSetFilterNdDescriptor(dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));

    // initialize weight and inputs
    // inputs
    generate_data(curand_gen, (float*)x, input_length);
    if (hx != nullptr) generate_data(curand_gen, (float*)hx, hidden_length);
    if (cx != nullptr) generate_data(curand_gen, (float*)cx, hidden_length);

    generate_data(curand_gen, (float*)dy, output_length);
    if (dhy != nullptr) generate_data(curand_gen, (float*)dhy, hidden_length);
    if (dcy != nullptr) generate_data(curand_gen, (float*)dcy, hidden_length);

    // weights
    int num_linear_layers = 0;
    switch (rnn_mode) {
        case CUDNN_RNN_RELU:
        case CUDNN_RNN_TANH:
            num_linear_layers = 2;
            break;
        case CUDNN_LSTM:
            num_linear_layers = 8;
            break;
        case CUDNN_GRU:
            num_linear_layers = 6;
            break;
    }

    for (int layer = 0; layer < num_layers; layer++) {
        cudnnDataType_t data_type;
        cudnnTensorFormat_t format;
        int nb_dim, filter_dim[3];
        cudnnFilterDescriptor_t linear_filter_desc, linear_bias_desc;
        float *linear_layer_filter, *linear_bias;

        for (int linear_layer = 0; linear_layer < num_linear_layers; linear_layer++) {
            // filter
            checkCudnnErrors(cudnnCreateFilterDescriptor(&linear_filter_desc));
            checkCudnnErrors(cudnnGetRNNLinLayerMatrixParams(cudnnHandle, 
                                                        rnn_desc, layer, x_desc[0], w_desc, weights, linear_layer, linear_filter_desc, (void**)&linear_layer_filter));
            checkCudnnErrors(cudnnGetFilterNdDescriptor(linear_filter_desc,
                                                        3, &data_type, &format, &nb_dim, filter_dim));
            generate_data(curand_gen, linear_layer_filter, filter_dim[0] * filter_dim[1] * filter_dim[2]);

            // bias
            checkCudnnErrors(cudnnCreateFilterDescriptor(&linear_bias_desc));
            checkCudnnErrors(cudnnGetRNNLinLayerBiasParams(cudnnHandle,
                                                        rnn_desc, layer, x_desc[0], w_desc, weights, linear_layer, linear_bias_desc, (void**)&linear_bias));
            checkCudnnErrors(cudnnGetFilterNdDescriptor(linear_bias_desc, 3, &data_type, &format, &nb_dim, filter_dim));
            generate_data(curand_gen, linear_bias, filter_dim[0] * filter_dim[1] * filter_dim[2]);

            checkCudnnErrors(cudnnDestroyFilterDescriptor(linear_filter_desc));
            checkCudnnErrors(cudnnDestroyFilterDescriptor(linear_bias_desc));
        }
    }

    /* Dynamic persistent RNN plan (if using this algorithm)*/
    cudnnPersistentRNNPlan_t rnn_plan;
    if (rnn_algo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
        checkCudnnErrors(cudnnCreatePersistentRNNPlan(rnn_desc, batch_size, CUDNN_DATA_FLOAT, &rnn_plan));
        checkCudnnErrors(cudnnSetPersistentRNNPlan(rnn_desc, rnn_plan));
    }

    // RUN RNN
    checkCudaErrors(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    float time_forward, time_backward1, time_backward2;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));
    checkCudnnErrors(cudnnRNNForwardTraining(cudnnHandle, rnn_desc, seq_length,
                                            x_desc, x,
                                            hx_desc, hx,
                                            cx_desc, cx,
                                            w_desc, weights,
                                            y_desc, y,
                                            hy_desc, hy,
                                            cy_desc, cy,
                                            workspace, workspace_size,
                                            reserved_space, reserved_size));
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_forward, start, stop));

    checkCudaErrors(cudaEventRecord(start));
    checkCudnnErrors(cudnnRNNBackwardData(cudnnHandle, rnn_desc, seq_length,
                                        y_desc, y,
                                        dy_desc, dy,
                                        dhy_desc, dhy,
                                        dcy_desc, dcy,
                                        w_desc, weights,
                                        hx_desc, hx,
                                        cx_desc, cx,
                                        dx_desc, dx,
                                        dhx_desc, dhx,
                                        dcx_desc, dcx,
                                        workspace, workspace_size,
                                        reserved_space, reserved_size));
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_backward1, start, stop));

    checkCudaErrors(cudaEventRecord(start));
    checkCudaErrors(cudaMemset(gweights, 0, weight_size));
    checkCudnnErrors(cudnnRNNBackwardWeights(cudnnHandle, rnn_desc, seq_length,
                                            x_desc, x, hx_desc, hx, y_desc, y,
                                            workspace, workspace_size,
                                            dw_desc, gweights,
                                            reserved_space, reserved_size));

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&time_backward2, start, stop));    

    // Calculate FLOPS
    printf("RNN Forward: %3.0f GFLOPS\n",  num_linear_layers * 2ull * (bidirectional ? 2 : 1) * input_size * hidden_size * seq_length * batch_size * num_layers / (1e6 * time_forward));
    // printf("Backward: %3.0f GFLOPS, ", num_linear_layers * 4ull * (bidirectional ? 2 : 1) * input_size * hidden_size * seq_length * batch_size * num_layers / (1e6 * (time_backward1 + time_backward2)));

    /* Destroy handles and resources */
    if (rnn_algo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC)
        cudnnDestroyPersistentRNNPlan(rnn_plan);

    cudaFree(x);
    cudaFree(hx);
    cudaFree(cx);
    cudaFree(y);
    cudaFree(hy);
    cudaFree(cy);
    cudaFree(dx);
    cudaFree(dhx);
    cudaFree(dcx);
    cudaFree(dy);
    cudaFree(dhy);
    cudaFree(dcy);
    cudaFree(workspace);
    cudaFree(reserved_space);
    cudaFree(weights);
    cudaFree(gweights);
    cudaFree(state);

    for (int i = 0; i < seq_length; i++)
    {
        cudnnDestroyTensorDescriptor(x_desc[i]);
        cudnnDestroyTensorDescriptor(y_desc[i]);

        cudnnDestroyTensorDescriptor(dx_desc[i]);
        cudnnDestroyTensorDescriptor(dy_desc[i]);
    }

    cudnnDestroyTensorDescriptor(hx_desc);
    cudnnDestroyTensorDescriptor(cx_desc);
    cudnnDestroyTensorDescriptor(hy_desc);
    cudnnDestroyTensorDescriptor(cy_desc);

    cudnnDestroyTensorDescriptor(dhx_desc);
    cudnnDestroyTensorDescriptor(dcx_desc);
    cudnnDestroyTensorDescriptor(dhy_desc);
    cudnnDestroyTensorDescriptor(dcy_desc);

    cudnnDestroyDropoutDescriptor(dropout_desc);
    cudnnDestroyRNNDescriptor(rnn_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyFilterDescriptor(dw_desc);

    cudnnDestroy(cudnnHandle);
    curandDestroyGenerator(curand_gen);
}

void cublas_operation(int rnn_mode, unsigned long long num_ops, int input_size, int hidden_size, int seq_length, int batch_size, int num_layers)
{
    float *input_weight, *recurrent_weight;
    float *x, *y, *h;
    float ms;

    // we will emulate RNN operation with two SGEMM operation, so we can reduce the number operation per layer
    int num_linear_layers = 0;
    switch (rnn_mode) {
        case CUDNN_RNN_RELU:
        case CUDNN_RNN_TANH:
            num_linear_layers = 1;
            break;
        case CUDNN_LSTM:
            num_linear_layers = 4;
            break;
        case CUDNN_GRU:
            num_linear_layers = 3;
            break;
    }

    checkCudaErrors(cudaMalloc((void**)&input_weight, input_size * hidden_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&recurrent_weight, hidden_size * hidden_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&x, batch_size * input_size * seq_length * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&y, batch_size * hidden_size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&h, batch_size * hidden_size * sizeof(float)));

    // create cublas handle
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // create cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // generate input data
    curandGenerator_t curand_gen;
    unsigned long long seed = 2019UL;
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, seed);

    // initialize input data
    generate_data(curand_gen, x, batch_size * input_size * seq_length);
    generate_data(curand_gen, input_weight, input_size * hidden_size);
    generate_data(curand_gen, recurrent_weight, hidden_size * hidden_size);

    float alpha = 1.f, beta = 0.f;

    cudaEventRecord(start);
    for (int layer = 0; layer < num_layers; layer++)
    {
        for (int linear_layer = 0; linear_layer < num_linear_layers; linear_layer++)
        {
            for (int sequence = 0; sequence < seq_length; sequence++)
            {
                cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            hidden_size, input_size, batch_size,
                            &alpha, input_weight, input_size, x, input_size, 
                            &beta, h, hidden_size);
        
                cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            hidden_size, hidden_size, batch_size,
                            &alpha, recurrent_weight, hidden_size, h, hidden_size,
                            &beta, y, hidden_size);
            }
        }
    }
    checkCudaErrors(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    // Calculate Flops
    printf("GEMM performance: %3.0f GFLOPS\n", num_linear_layers * num_ops * input_size * hidden_size * seq_length * batch_size * num_layers / (1e6 * ms));

    // destroy handles and memories
    cublasDestroy(cublas_handle);
    curandDestroyGenerator(curand_gen);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(x);
    cudaFree(y);
    cudaFree(h);
    cudaFree(input_weight);
    cudaFree(recurrent_weight);
}

void generate_data(const curandGenerator_t generator, float *data, int length)
{
    curandGenerateNormal(generator, data, length, 0.f, 1.f);
}

int main()
{
    // configuration rnn
    int seq_length = 512;
    int num_layers = 4;
    int hidden_size = 512;
    int input_size = hidden_size;
    int batch_size = 32;
    float dropout_rate = 0;
    bool bidirectional = 0;
    int mode = 2; // LSTM
    int persistent = 2;

    for (int step = 1; step <= 8; step++)
    {
        batch_size = 32 * step;
        printf("Batch Size: %3d\n", batch_size);
        rnn_operation(seq_length, num_layers, hidden_size, input_size, batch_size, dropout_rate, bidirectional, mode, persistent);
        cublas_operation(mode, 2ull, input_size, hidden_size, seq_length, batch_size, num_layers);
    }
}
