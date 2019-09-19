#ifndef _REDUCTION_H_
#define _REDUCTION_H_

// @reduction_kernel.cu
void reduction(float *d_out, float *d_in, int n_threads, int size);

// @naive_reduction_kernel.cu
void global_reduction(float *d_out, float *d_in, int n_threads, int size);
// void atomic_reduction(float *d_out, float *d_in, int n_threads, int size);

#endif // _REDUCTION_H_
