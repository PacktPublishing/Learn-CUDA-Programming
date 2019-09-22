#include<stdio.h>
#include"scrImagePgmPpmPackage.h"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

//Step 1: Declare the texture memory
texture<unsigned char, 2, cudaReadModeElementType> tex;


//Kernel which calculate the resized image
__global__ void calculateHistogram(unsigned int *imageHistogram, unsigned int width, unsigned int height, cudaTextureObject_t texObj)
{
        const unsigned int tidX = blockIdx.x*blockDim.x + threadIdx.x;
        const unsigned int tidY = blockIdx.y*blockDim.y + threadIdx.y;
	
	const unsigned int localId = threadIdx.y*blockDim.x+threadIdx.x;
	const unsigned int histStartIndex = (blockIdx.y*gridDim.x+blockIdx.x) * 256;

	__shared__ unsigned int histo_private[256];

	if(localId <256)
		histo_private[localId] = 0;
	__syncthreads();

	unsigned char imageData =  tex2D<unsigned char>(texObj,(float)(tidX),(float)(tidY));
	atomicAdd(&(histo_private[imageData]), 1);

	 __syncthreads();

	if(localId <256)
        	imageHistogram[histStartIndex+localId] = histo_private[localId];      
       	
}


int main(int argc, char*argv[])
{
	int height=0, width =0, noOfHistogram=0;
	//Define the scaling ratio	
	unsigned char*data;
	unsigned int *imageHistogram, *d_imageHistogram;

	char inputStr[1024] = {"aerosmith-double.pgm"};
	cudaError_t returnValue;

	//Create a channel Description to be used while linking to the tecture
	cudaArray* cu_array;
	cudaChannelFormatKind kind = cudaChannelFormatKindUnsigned;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, kind);


	get_PgmPpmParams(inputStr, &height, &width);	//getting height and width of the current image
	data = (unsigned char*)malloc(height*width*sizeof(unsigned char));
	printf("\n Reading image width height and width [%d][%d]", height, width);
	scr_read_pgm( inputStr , data, height, width );//loading an image to "inputimage"
	//height=64;width=64;
	//data = (unsigned char*)malloc(height*width*sizeof(unsigned char));
	/*for(int i=0;i<64;i++)
	{
		for(int j=0;j<64;j++)
		{
			if(i<32)  data[i*width+j] = 22;
			else data[i*width+j] = 255;
		}
	}*/
			

	// One istogram per image block . Since it is char image max size range ie between 0-255 
	noOfHistogram = (height/BLOCK_SIZE_Y) * (width/BLOCK_SIZE_X) * 256;
	imageHistogram = (unsigned int*)malloc(noOfHistogram*sizeof(unsigned int));

	//Allocate CUDA Array
 	returnValue = cudaMallocArray( &cu_array, &channelDesc, width, height);
	returnValue = (cudaError_t)(returnValue | cudaMemcpyToArray( cu_array, 0, 0, data, height*width*sizeof(unsigned char), cudaMemcpyHostToDevice));

        if(returnValue != cudaSuccess) printf("\n Got error while running CUDA API Array Copy");

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cu_array;
	//Step 2 Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	// Create texture object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

        if(returnValue != cudaSuccess) printf("\n Got error while running CUDA API Bind Texture");
	cudaMalloc(&d_imageHistogram, noOfHistogram*sizeof(unsigned int) );

        dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y,1);
        dim3 dimGrid(width/dimBlock.x,height/dimBlock.y,1);
	printf("\n Launching grid with blocks [%d][%d] ", dimGrid.x,dimGrid.y);

        calculateHistogram<<<dimGrid, dimBlock>>>(d_imageHistogram,width,height, texObj);

        returnValue = (cudaError_t)(returnValue | cudaThreadSynchronize());

	returnValue = (cudaError_t)(returnValue |cudaMemcpy (imageHistogram , d_imageHistogram, noOfHistogram*sizeof(unsigned int), cudaMemcpyDeviceToHost ));
        if(returnValue != cudaSuccess) printf("\n Got error while running CUDA API kernel");

	// Step 4: Destroy texture object
	cudaDestroyTextureObject(texObj);

	printf("\n Histogram perr section is as follows: ");
	for(int i=0;i< noOfHistogram/256;i++)
	{
		printf("\n----------------------------- Histograam for block %d----------------------- \n", i);
		for(int j=0;j<256;j++)
		{
			int index = i*256 + j;
			printf( "[%d=[%d]]  ", j, imageHistogram[index]);
		}
	}	
		
	if(data != NULL)
		free(data);
	if(cu_array !=NULL)
		cudaFreeArray(cu_array);
	if(imageHistogram != NULL)
		free(imageHistogram);
	if(d_imageHistogram!=NULL)
		cudaFree(d_imageHistogram);
	
	return 0;
}
