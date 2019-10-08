#ifndef _REDUCTION_H_
#define _REDUCTION_H_

// @reduction_wrp_atmc_kernel.cu
// @reduction_blk_atmc_kernel.cu
// @reduction_kernel.cu
void atomic_reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads);

#endif // _REDUCTION_H_