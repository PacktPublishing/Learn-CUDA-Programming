#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32 //Warp Size 
#define LOOPS 1000

#define MAX_BIT_POS 31
#define MIN_BIT_POS 0

__device__ unsigned int d_data[WARP_SIZE];

//Implements Radix sort using warp level primitives
__global__ void radix_warp_sort(){

  __shared__ volatile unsigned int s_data[WARP_SIZE*2];
  
  // load data from global memory into shared array in coalesced manner
  s_data[threadIdx.x] = d_data[threadIdx.x];
  unsigned int offset  = 0;
  unsigned int pos =0;
  unsigned int bit_mask=  1<<MIN_BIT_POS;
  unsigned int thread_mask = 0xFFFFFFFFU << threadIdx.x;
  //  for each LSB to MSB
  for (int i = MIN_BIT_POS; i <= MAX_BIT_POS; i++){

    unsigned int data = s_data[((WARP_SIZE-1)-threadIdx.x)+offset];
    unsigned int bit  = data&bit_mask;

    // get population of ones and zeroes
    unsigned int active = __activemask();
    unsigned int ones = __ballot_sync(active,bit); 
    unsigned int zeroes = ~ones;

//switch ping-pong buffers
    offset ^= WARP_SIZE; // switch ping-pong buffers
    // do zeroes, then ones
    if (!bit) // threads with a zero bit
      // get my position in ping-pong buffer
      pos = __popc(zeroes&thread_mask);
    else        // threads with a one bit
      // get my position in ping-pong buffer
      pos = __popc(zeroes)+__popc(ones&thread_mask);
    
    // move to buffer 
    s_data[pos-1+offset] = data;
    
    // repeat for next bit
    bit_mask <<= 1;
    }
  // save results to global memory
  d_data[threadIdx.x] = s_data[threadIdx.x+offset];
  }

void validate (unsigned int*h_data)
{
    for (int i = 0; i < WARP_SIZE-1; i++)
      if (h_data[i] > h_data[i+1])
        {
          printf("Sorting uncessfull: h_data[%d] = %d > h_data[%d] = %d\n", i, h_data[i],i+1, h_data[i+1]); 
          exit(0);
        }
}

int  main(){

  unsigned int h_data[WARP_SIZE];
  unsigned int size = WARP_SIZE*sizeof(unsigned int);
  for (int count = 0; count < LOOPS; count++){
    for (int i = 0; i < WARP_SIZE; i++) 
      h_data[i] = rand()%(1U<<MAX_BIT_POS);
    cudaMemcpyToSymbol(d_data, h_data, size);
    radix_warp_sort<<<1, WARP_SIZE>>>();
    cudaMemcpyFromSymbol(h_data, d_data, size);
    validate(h_data);
    }
  printf("Sorting Successful!\n");

  return 0;
}
