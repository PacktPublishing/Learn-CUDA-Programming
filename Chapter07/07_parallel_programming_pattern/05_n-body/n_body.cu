#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "n_body.h"

__global__ void calculateBodyForce(float4 *p, float4 *v, float dt, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
  	if (i < n) {
    		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    		for (int tile = 0; tile < gridDim.x; tile++) {
      			__shared__ float3 shared_position[BLOCK_SIZE];
      			float4 temp_position = p[tile * blockDim.x + threadIdx.x];
      			shared_position[threadIdx.x] = make_float3(temp_position.x, temp_position.y, temp_position.z);
      			__syncthreads(); //synchronoze to make sure all tile data is available in shared memory

      			for (int j = 0; j < BLOCK_SIZE; j++) {
        			float dx = shared_position[j].x - p[i].x;
        			float dy = shared_position[j].y - p[i].y;
        			float dz = shared_position[j].z - p[i].z;
        			float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        			float invDist = rsqrtf(distSqr);
        			float invDist3 = invDist * invDist * invDist;

        			Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      			}
      			__syncthreads(); // synchrnize before looping to other time
    		} //tile loop ends here		

    		v[i].x += dt*Fx; v[i].y += dt*Fy; v[i].z += dt*Fz;
  	} //if ends here
}

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  
  const float dt = 0.01f; // time step
  const int nIters = 100;  // simulation iterations
  
  int size = 2*nBodies*sizeof(float4);
  float *buf = (float*)malloc(size);
  NBodySystem p = { (float4*)buf, ((float4*)buf) + nBodies };

  generateRandomizeBodies(buf, 8*nBodies); // Init pos / vel data

  float *d_buf;
  cudaMalloc(&d_buf, size);
  NBodySystem d_p = { (float4*)d_buf, ((float4*)d_buf) + nBodies };

  int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int iter = 1; iter <= nIters; iter++) {

    cudaMemcpy(d_buf, buf, size, cudaMemcpyHostToDevice);
    calculateBodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p.pos, d_p.vel, dt, nBodies);
    cudaMemcpy(buf, d_buf, size, cudaMemcpyDeviceToHost);

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p.pos[i].x += p.vel[i].x*dt;
      p.pos[i].y += p.vel[i].y*dt;
      p.pos[i].z += p.vel[i].z*dt;
    }

    printf("Iteration %d\n", iter);
  }

  free(buf);
  cudaFree(d_buf);
}

void generateRandomizeBodies(float *data, int n) {
	
	float max = (float)RAND_MAX;
        for (int i = 0; i < n; i++) {
                data[i] = 2.0f * (rand() / max) - 1.0f;
  }
}

