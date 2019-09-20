#include<stdio.h>
#include<stdlib.h>
#include"scrImagePgmPpmPackage.h"
#include<omp.h> 

#define MIN(X,Y) ((X<Y) ? X:Y)

typedef void (*exitroutinetype)(char *err_msg);
extern void acc_set_error_routine(exitroutinetype callback_routine);

void handle_gpu_errors(char *err_msg) {
  printf("GPU Error: %s", err_msg);
  printf("Exiting...\n\n");
  exit(-1);
}

void merge_serial(unsigned char *in1, unsigned char*in2, unsigned char *out, long w, long h)
{
  long x, y;
  for(y = 0; y < h; y++) {
    for(x = 0; x < w; x++) {
      out[y * w + x]      = (in1[y * w + x]+in2[y * w + x])/2;
    }
  }
}
void merge_parallel_pragma(unsigned char *in1, unsigned char*in2,unsigned char *out, long w, long h)
{
  long x, y;
#pragma acc parallel loop gang copyin(in1[:h*w],in2[:h*w]) copyout(out[:h*w]) 
  for(y = 0; y < h; y++) {
#pragma acc loop vector
    for(x = 0; x < w; x++) {
	out[y * w + x]      = (in1[y * w + x]+in2[y * w + x])/2;
    }
  }
}

void merge_data_blocked(unsigned char *in1, unsigned char*in2,unsigned char *out, long w, long h)
{
  long x, y;

#pragma acc enter data create(out[:w*h])
#pragma acc enter data copyin(in1[:w*h],in2[:h*w])
double st = omp_get_wtime();
  const long numBlocks = 8;
  const long rowsPerBlock = (h+(numBlocks-1))/numBlocks;
  for(long block = 0; block < numBlocks; block++) {
    long lower = block*rowsPerBlock;
    long upper = MIN(h, lower+rowsPerBlock);
#pragma acc parallel loop gang present(in1, in2,out)
    for(y = lower; y < upper; y++) {
#pragma acc loop vector
      for(x = 0; x < w; x++) {
        out[y * w + x]      = (in1[y * w + x]+in2[y * w + x])/2;
      }
    }
  }
 printf("Time taken for OpenACC merge(kernel only) with Blocking: %.4f seconds\n", omp_get_wtime()-st);
#pragma acc exit data delete(in1,in2) copyout(out[:w*h])
}

void merge_async_pipelined(unsigned char *in1, unsigned char*in2,unsigned char *out, long w, long h)
{
  long x, y;

#pragma acc enter data create(in1[:w*h], in2[:h*w], out[:w*h])
  const long numBlocks = 8;
  const long rowsPerBlock = (h+(numBlocks-1))/numBlocks;
  for(long block = 0; block < numBlocks; block++) {
    long lower = block*rowsPerBlock; // Compute Lower
    long upper = MIN(h, lower+rowsPerBlock); // Compute Upper
#pragma acc update device(in1[lower*w:(upper-lower)*w],in2[lower*w:(upper-lower)*w]) async(block%2)
#pragma acc parallel loop present(in1,in2, out) async(block%2)
    for(y = lower; y < upper; y++) {
#pragma acc loop
      for(x = 0; x < w; x++) {
        out[y * w + x]      = (in1[y * w + x]+in2[y * w + x])/2;
      }
    }
#pragma acc update self(out[lower*w:(upper-lower)*w]) async(block%2)
  }
#pragma acc wait
#pragma acc exit data delete(in1, in2, out)
}

int main(int argc, char*argv[])
{
	int height=0, width =0;
	
	unsigned char*data1,*data2;
	unsigned char*merged_data;

	char inputStr1[1024] = {"cat.pgm"};
	char inputStr2[1024] = {"dog.pgm"};
	char outputSerialStr[1024] = {"merged_serial.pgm"};
	char outputPragmaStr[1024] = {"merged_parallel.pgm"};
	char outputDataStr[1024] = {"merged_data.pgm"};
	char outputPipelineStr[1024] = {"merged_pipeline.pgm"};


	get_PgmPpmParams(inputStr1, &height, &width);	//getting height and width of the current image
	data1 = (unsigned char*)malloc((long)(height*width*sizeof(unsigned char)));
	data2 = (unsigned char*)malloc((long)(height*width*sizeof(unsigned char)));
	merged_data = (unsigned char*)malloc((long)(height*width*sizeof(unsigned char)));
	printf("\n Reading image width height and width [%d][%d]\n", height, width);
	scr_read_pgm( inputStr1 , data1, height, width );//loading an image to "inputimage"
	scr_read_pgm( inputStr2 , data2, height, width );//loading an image to "inputimage"
	//acc_set_error_routine(&handle_gpu_errors);	

	// Warm up call to get right numbers for timings
	merge_async_pipelined(data1,data2,merged_data, width,height);	
	
	double st = omp_get_wtime();	
	merge_serial(data1,data2,merged_data, width,height);
	printf("Time taken for serial merge: %.4f seconds\n", omp_get_wtime()-st);
	scr_write_pgm( outputSerialStr, merged_data, height, width, "Merged Serial" ); //storing the image with the detections

	st = omp_get_wtime();	
	merge_parallel_pragma(data1,data2,merged_data, width,height);	
	printf("Time taken for OpenACC merge(data+kernel): %.4f seconds\n", omp_get_wtime()-st);
	scr_write_pgm( outputPragmaStr, merged_data, height, width, "Merged OpenACC" ); //storing the image with the detections

	st = omp_get_wtime();	
	merge_data_blocked(data1,data2,merged_data, width,height);	
	printf("	Time taken for OpenACC merge(data _kernel) with blocking: %.4f seconds\n", omp_get_wtime()-st);
	scr_write_pgm( outputDataStr, merged_data, height, width, "Merged Data Blocked" ); //storing the image with the detections

	st = omp_get_wtime();	
	merge_async_pipelined(data1,data2,merged_data, width,height);	
	printf("Time taken for OpenACC merge (data+kernel)with Pipeline Async: %.4f seconds\n", omp_get_wtime()-st);
	scr_write_pgm( outputPipelineStr, merged_data, height, width, "Merged Pipeline" ); //storing the image with the detections
		
	if(data1 != NULL)
		free(data1);
	if(data2 != NULL)
		free(data2);
	if(merged_data != NULL)
		free(merged_data);
	printf("\n Done");
	return 0;
}
