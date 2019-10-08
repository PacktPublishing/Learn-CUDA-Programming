#ifndef _HELPER_CU_H_
#define _HELPER_CU_H_

#include <curand.h>
#include <cufft.h>
#include "fp16.cuh"

namespace op {
template <typename T>
typename std::enable_if<std::is_same<T, float>::value>::type
curand(curandGenerator_t generator,
            T *buffer,
            size_t length)
{
    curandGenerateUniform(generator, buffer, length);
}

void FloatToComplex(cufftComplex *complex, const float *real, const float *imag, const size_t length);

template <typename T>
typename std::enable_if<std::is_same<T, cufftComplex>::value>::type
curand(curandGenerator_t generator,
            T *buffer,
            size_t length)
{
    float *buffer_fp32;

    cudaMalloc((void **)&buffer_fp32, length * sizeof(float));
    curandGenerateUniform(generator, buffer_fp32, length);

    // convert generated real data into complex type
    FloatToComplex(buffer, buffer_fp32, nullptr, length);
    cudaFree(buffer_fp32);
}

template <typename T>
typename std::enable_if<std::is_same<T, half>::value>::type
curand(curandGenerator_t generator,
            T *buffer,
            size_t length)
{
    float *buffer_fp32;

    cudaMalloc((void **)&buffer_fp32, length * sizeof(float));
    curandGenerateUniform(generator, buffer_fp32, length);

    // convert generated single floating to half floating
    fp16::float2half(buffer, buffer_fp32, length);
    cudaFree(buffer_fp32);
}
}

#endif // _HELPER_CU_H_