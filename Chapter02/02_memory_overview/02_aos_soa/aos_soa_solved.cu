#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<string.h>


#define NUM_THREADS 256

#define IMG_SIZE 1048576

struct Coefficients_AOS {
  int* r;
  int* b;
  int* g;
  int* hue;
  int* saturation;
  int* maxVal;
  int* minVal;
  int* finalVal; 
};


__global__
void complicatedCalculation(Coefficients_AOS  data)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;


  int grayscale = (data.r[i] + data.g[i] + data.b[i])/data.maxVal[i];
  int hue_sat = data.hue[i] * data.saturation[i] / data.minVal[i];
  data.finalVal[i] = grayscale*hue_sat; 
}

void complicatedCalculation()
{

  Coefficients_AOS d_x;

  cudaMalloc(&d_x.r, IMG_SIZE*sizeof(int)); 
  cudaMalloc(&d_x.g, IMG_SIZE*sizeof(int)); 
  cudaMalloc(&d_x.b, IMG_SIZE*sizeof(int)); 
  cudaMalloc(&d_x.hue, IMG_SIZE*sizeof(int)); 
  cudaMalloc(&d_x.saturation, IMG_SIZE*sizeof(int)); 
  cudaMalloc(&d_x.maxVal, IMG_SIZE*sizeof(int)); 
  cudaMalloc(&d_x.minVal, IMG_SIZE*sizeof(int)); 
  cudaMalloc(&d_x.finalVal, IMG_SIZE*sizeof(int)); 


  int num_blocks = IMG_SIZE/NUM_THREADS;

  complicatedCalculation<<<num_blocks,NUM_THREADS>>>(d_x);

  
  cudaFree(d_x.r);
  cudaFree(d_x.g);
  cudaFree(d_x.b);
  cudaFree(d_x.hue);
  cudaFree(d_x.saturation);
  cudaFree(d_x.maxVal);
  cudaFree(d_x.maxVal);
  cudaFree(d_x.minVal);
  cudaFree(d_x.finalVal);
}



int main(int argc, char*argv[])
{

	complicatedCalculation();
	return 0;
}






