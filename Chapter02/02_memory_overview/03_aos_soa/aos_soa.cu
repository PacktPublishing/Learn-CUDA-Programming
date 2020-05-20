#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<string.h>


#define NUM_THREADS 256

#define IMG_SIZE 1048576

struct Coefficients_SOA {
  int r;
  int b;
  int g;
  int hue;
  int saturation;
  int maxVal;
  int minVal;
  int finalVal; 
};


__global__
void complicatedCalculation(Coefficients_SOA*  data)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;


  int grayscale = (data[i].r + data[i].g + data[i].b)/data[i].maxVal;
  int hue_sat = data[i].hue * data[i].saturation / data[i].minVal;
  data[i].finalVal = grayscale*hue_sat; 
}

void complicatedCalculation()
{

  Coefficients_SOA* d_x;

  cudaMalloc(&d_x, IMG_SIZE*sizeof(Coefficients_SOA)); 

  int num_blocks = IMG_SIZE/NUM_THREADS;

  complicatedCalculation<<<num_blocks,NUM_THREADS>>>(d_x);

  cudaFree(d_x);
}



int main(int argc, char*argv[])
{

	complicatedCalculation();
	return 0;
}






