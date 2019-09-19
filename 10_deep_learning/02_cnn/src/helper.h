#ifndef _HELPER_H_
#define _HELPER_H_

#include <cudnn.h>
#include <cublas_v2.h>

// #include <helper_cuda.h>
#include <curand.h>

namespace cudl
{
#define BLOCK_DIM_1D    512
#define BLOCK_DIM       16

/* DEBUG FLAGS */
#define DEBUG_FORWARD   0
#define DEBUG_BACKWARD  0

#define DEBUG_CONV      0
#define DEBUG_DENSE     0
#define DEBUG_SOFTMAX   0
#define DEBUG_UPDATE    0

#define DEBUG_LOSS      0
#define DEBUG_ACCURACY  0

/* CUDA API error return checker */
#ifndef checkCudaErrors
#define checkCudaErrors(err)                                                                        \
    {                                                                                               \
        if (err != cudaSuccess)                                                                     \
        {                                                                                           \
            fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    err, cudaGetErrorString(err), __FILE__, __LINE__);                              \
			fprintf(stderr, "%d\n", cudaSuccess);													\
            exit(-1);                                                                               \
        }                                                                                           \
    }
#endif

static const char *_cublasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}

#define checkCublasErrors(err)                                                                        \
    {                                                                                                 \
        if (err != CUBLAS_STATUS_SUCCESS)                                                             \
        {                                                                                             \
            fprintf(stderr, "checkCublasErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    err, _cublasGetErrorEnum(err), __FILE__, __LINE__);                                 \
            exit(-1);                                                                                 \
        }                                                                                             \
    }

#define checkCudnnErrors(err)                                                                        \
    {                                                                                                \
        if (err != CUDNN_STATUS_SUCCESS)                                                             \
        {                                                                                            \
            fprintf(stderr, "checkCudnnErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    err, cudnnGetErrorString(err), __FILE__, __LINE__);                              \
            exit(-1);                                                                                \
        }                                                                                            \
    }

// cuRAND API errors
static const char *_curandGetErrorEnum(curandStatus_t error) {
    switch (error) {
    case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";

    case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";

    case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";

    case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";

    case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";

    case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";

    case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";

    case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";

    case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";

    case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}


#define checkCurandErrors(err)                                                                        \
    {                                                                                                \
        if (err != CURAND_STATUS_SUCCESS)                                                             \
        {                                                                                            \
            fprintf(stderr, "checkCurandErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
                    err, _curandGetErrorEnum(err), __FILE__, __LINE__);                              \
            exit(-1);                                                                                \
        }                                                                                            \
    }

// container for cuda resources
class CudaContext
{
    public:
    CudaContext()
    {
        cublasCreate(&_cublas_handle);
        checkCudaErrors(cudaGetLastError());
        checkCudnnErrors(cudnnCreate(&_cudnn_handle));
    }
    ~CudaContext()
    {
        cublasDestroy(_cublas_handle);
        checkCudnnErrors(cudnnDestroy(_cudnn_handle));
    }

    cublasHandle_t cublas() { 
        //std::cout << "Get cublas request" << std::endl; getchar();
        return _cublas_handle; };
    cudnnHandle_t cudnn() { return _cudnn_handle; };

    const float one       =  1.f;
    const float zero      =  0.f;
    const float minus_one = -1.f;

    private:
    cublasHandle_t _cublas_handle;
    cudnnHandle_t  _cudnn_handle;
};

} // namespace cudl

#endif // _HELPER_H_