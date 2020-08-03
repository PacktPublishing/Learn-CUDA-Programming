#include "gaussian_multi_gpu_rdma.h"

void _report(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if(result)
    {
        fprintf(stderr, "CUDA error at %s:%d code = %d (%s) \"%s\" \n",
                file, line, result, cudaGetErrorString(result), func);
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(-1);
    }
}

    int setupPeerToPeer(int GPUCount)
    {
        
        int canAccessPeer;
        for(int i = 0; i < GPUCount; i++)
        {
            checkCudaErrors( cudaSetDevice(i) );
        }

        for(int i = 0; i < GPUCount; i++)
        {
            checkCudaErrors( cudaSetDevice(i) );
            for(int j = 0; j < GPUCount; j++)
            {
                if(i == j) continue;
                checkCudaErrors( cudaDeviceCanAccessPeer(&canAccessPeer, i, j) );
                if(canAccessPeer)
                {
                    printf("Can access memory of device %d from device %d\n", j, i);
                    checkCudaErrors( cudaDeviceEnablePeerAccess(j, 0) );
                }    
                else
                {
                     printf("Can not access memory of device %d from device %d\n", j, i);
                    return 0;
                }

            }
        }
        return 1;
    }

    int testPeerToPeer(int GPUCount)
    {
        char** buffer;

        int buffersize = 1024 * sizeof(char);
        buffer = (char**) malloc(GPUCount * sizeof(char*));

        for(int i = 0; i < GPUCount; i++)
        {
            checkCudaErrors( cudaSetDevice(i) );
            checkCudaErrors( cudaMalloc((void**)&buffer[i], buffersize) );
        }

        for(int i = 0; i < GPUCount; i++)
        {
            for(int j = 0; j < GPUCount; j++)
            {
                if(i == j) continue;
                checkCudaErrors( cudaMemcpyPeer(buffer[i], i, buffer[j], j, buffersize) );
            }
        }

        for(int i = 0; i < GPUCount; i++)
        {
            checkCudaErrors( cudaFree(buffer[i]) );
        }
        return 1;
    }
