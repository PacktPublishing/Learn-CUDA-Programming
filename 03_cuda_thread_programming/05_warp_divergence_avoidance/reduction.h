#ifndef _REDUCTION_H_
#define _REDUCTION_H_

// @reduction_kernel_1.cu
int reduction_1(float *g_outPtr, float *g_inPtr, int size, int n_threads);

// @reduction_krenel_2.cu
int reduction_2(float *g_outPtr, float *g_inPtr, int size, int n_threads);

#define max(a, b) (a) > (b) ? (a) : (b)

#endif // _REDUCTION_H_