#ifndef _REDUCTION_H_
#define _REDUCTION_H_

// @ calling the reduction kernel
int reduction(float *g_outPtr, float *g_inPtr, int size, int n_threads);

#define max(a, b) (a) > (b) ? (a) : (b)

#endif // _REDUCTION_H_