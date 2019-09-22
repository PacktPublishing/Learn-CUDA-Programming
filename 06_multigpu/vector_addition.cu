

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < len)
    out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv) {
  int inputLength  = 1<<28;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;


  /*hostInput1 = (float*) malloc (sizeof(float) * inputLength);
  hostInput2 = (float*) malloc (sizeof(float) * inputLength);
  hostOutput = (float*) malloc (sizeof(float) * inputLength);a*/
  cudaMallocHost(&hostInput1, inputLength*sizeof(float));
  cudaMallocHost(&hostInput2, inputLength*sizeof(float));
  cudaMallocHost(&hostOutput, inputLength*sizeof(float));

  for(int i=0;i<inputLength;i++) {
    hostInput1[i] = i%1024;
    hostInput2[i] = i%1024;
  }
  
  cudaStream_t stream[4];
  float *d_A[4], *d_B[4], *d_C[4];
  int i, k, Seglen = 16384;
  int Gridlen = (Seglen - 1) / 256 + 1;
  
  for (i = 0; i < 4; i++) {
    cudaStreamCreateWithFlags(&stream[i],cudaStreamNonBlocking);
    cudaMalloc((void **)&d_A[i], Seglen * sizeof(float));
    cudaMalloc((void **)&d_B[i], Seglen * sizeof(float));
    cudaMalloc((void **)&d_C[i], Seglen * sizeof(float));
  }

  for (i = 0; i < inputLength; i += Seglen * 4) {
    for (k = 0; k < 4; k++) {
     
      cudaMemcpyAsync(d_A[k], hostInput1 + i + k * Seglen,
                      Seglen * sizeof(float), cudaMemcpyHostToDevice,
                      stream[k]);
      cudaMemcpyAsync(d_B[k], hostInput2 + i + k * Seglen,
                      Seglen * sizeof(float), cudaMemcpyHostToDevice,
                      stream[k]);
      vecAdd<<<Gridlen, 256, 0, stream[k]>>>(d_A[k], d_B[k], d_C[k],
                                             Seglen);
    }
    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);
    cudaStreamSynchronize(stream[2]);
    for (k = 0; k < 4; k++) {
      cudaMemcpyAsync(hostOutput + i + k * Seglen, d_C[k],
                      Seglen * sizeof(float), cudaMemcpyDeviceToHost,
                      stream[k]);
    }
  }
  cudaDeviceSynchronize();


  cudaFreeHost(hostInput1);
  cudaFreeHost(hostInput2);
  cudaFreeHost(hostOutput);
  
  for (k = 0; k < 3; k++) {
    cudaFree(d_A[k]);
    cudaFree(d_B[k]);
    cudaFree(d_C[k]);
  }

  return 0;
}

