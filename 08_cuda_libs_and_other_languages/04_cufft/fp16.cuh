#ifndef _FP16_CUH_
#define _FP16_CUH_

#include <cuda_fp16.h>

namespace fp16
{
void float2half(half *out, float *in, size_t length);
void half2float(float *out, half *in, size_t lenght);
}

#endif // _FP16_CUH_