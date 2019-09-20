#include<stdio.h>
#include<stdlib.h>
#include"scrImagePgmPpmPackage.h"
#include<omp.h> 

#define MIN(X,Y) ((X<Y) ? X:Y)

__global__
void merging_kernel(unsigned char *in1, unsigned char*in2,unsigned char *out, long w, long h, int lower, int upper)
{
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = (blockDim.y * blockIdx.y + threadIdx.y);
        int index = y*w+x;
        out[index] = (in1[index]+in2[index])/2;

}

void merge_single_gpu(unsigned char *in1, unsigned char*in2,unsigned char *out, long w, long h)
{
        unsigned char *d_in1,*d_in2,*d_out;
        dim3 blockDim(32,32,1);
        dim3 gridDim(w/32,h/32,1);
	int size = w*h*sizeof(unsigned char);
        cudaMalloc(&d_in1, size);
        cudaMalloc(&d_in2, size);
        cudaMalloc(&d_out, size);
	for(int i=0;i<32;i++) {
        cudaMemcpyAsync(d_in1, in1, size, cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_in2, in2, size, cudaMemcpyHostToDevice);
        merging_kernel<<<gridDim,blockDim>>>(d_in1,d_in2,d_out,w,h,0,h);
        cudaMemcpyAsync(out, d_out, size, cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
        cudaFree(d_in1);
        cudaFree(d_in2);
        cudaFree(d_out);
}


void merge_multi_gpu(unsigned char *in1, unsigned char*in2,unsigned char *out, long w, long h)
{
	int noDevices;
	cudaGetDeviceCount(&noDevices);
	cudaStream_t *streams;
	streams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * noDevices);
#pragma omp parallel num_threads(noDevices)
	{ 
		int block = omp_get_thread_num();
		int blockSize =  sizeof(unsigned char) * w* (h/noDevices);
		unsigned char *d_in1,*d_in2,*d_out;
		long lower = block*(h/noDevices); // Compute Lower
      		long upper = MIN(h, lower+(h/noDevices)); // Compute Upper
		dim3 blockDim(32,32,1);
        	dim3 gridDim(w/32,(h/noDevices)/32,1);

		cudaSetDevice(block);
        	cudaStreamCreate(&streams[block]);
		printf("\n T[%d] L[%d] U[%d] Gr[%d][%d]",block,lower,upper,gridDim.x,gridDim.y);
		cudaMalloc(&d_in1, blockSize);
		cudaMalloc(&d_in2, blockSize);
		cudaMalloc(&d_out, blockSize);
#pragma omp barrier
		for(int i=0;i<32;i++) {
			cudaMemcpyAsync(d_in1, in1+(lower*w), blockSize, cudaMemcpyHostToDevice,streams[block]);
			cudaMemcpyAsync(d_in2, in2+(lower*w), blockSize, cudaMemcpyHostToDevice, streams[block]);
			merging_kernel<<<gridDim,blockDim,0,streams[block]>>>(d_in1,d_in2,d_out,w,h,lower,upper);
			cudaMemcpyAsync(out+(lower*w), d_out, blockSize, cudaMemcpyDeviceToHost,streams[block]);
		}
		cudaFree(d_in1);
		cudaFree(d_in2);
		cudaFree(d_out);
	}
	cudaDeviceSynchronize();
}

int main(int argc, char*argv[])
{
	int height=0, width =0;
	
	unsigned char*data1,*data2;
	unsigned char*merged_data;

	char inputStr1[1024] = {"cat.pgm"};
	char inputStr2[1024] = {"dog.pgm"};
	char outputPipelineStr[1024] = {"merged_pipeline.pgm"};


	get_PgmPpmParams(inputStr1, &height, &width);	//getting height and width of the current image
	cudaMallocHost(&data1,height*width*sizeof(unsigned char));
	cudaMallocHost(&data2,height*width*sizeof(unsigned char));
	cudaMallocHost(&merged_data,height*width*sizeof(unsigned char));
	printf("\n Reading image  height and width [%d][%d]\n", height, width);
	scr_read_pgm( inputStr1 , data1, height, width );//loading an image to "inputimage"
	scr_read_pgm( inputStr2 , data2, height, width );//loading an image to "inputimage"

	merge_single_gpu(data1,data2,merged_data, width,height);	
	merge_multi_gpu(data1,data2,merged_data, width,height);	

	scr_write_pgm( outputPipelineStr, merged_data, height, width, "Merged Pipeline" ); //storing the image with the detections
		
	if(data1 != NULL)
		cudaFreeHost(data1);
	if(data2 != NULL)
		cudaFreeHost(data2);
	if(merged_data != NULL)
		cudaFreeHost(merged_data);
	printf("\n Done");
	return 0;
}
