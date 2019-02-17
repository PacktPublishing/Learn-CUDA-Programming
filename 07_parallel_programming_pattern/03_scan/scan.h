#include <cuda_runtime.h>
#include <stdarg.h>

#ifndef _UTILS_H_
#define _UTILS_H_

#define NUM_ITEM 16
#define BLOCK_DIM 8

#define DEBUG_OUTPUT_NUM 16

// generate input data
void generate_data(float *ptr, int length)
{
    // fill the buffer with random generated unsigned integers
    for (int i = 0; i < length; i++)
        ptr[i] = float(rand() - RAND_MAX / 2) / RAND_MAX;
}

bool validation(float *a, float *b, int length)
{
    float epsilon = 0.000001;
    bool result = true;
    for (int i = 0; i < length; i++)
        if (abs(a[i] - b[i]) >= epsilon)
            result = false;
    return result;
}

void print_val(float *h_list, int length, ...)
{
    va_list argptr;
    va_start(argptr, length);

    printf("%s\t", va_arg(argptr, char *));
    for (int i = 0; i < length; i++)
        printf("%.3f\t", h_list[i]);
    printf("\n");
}

#endif  // _UTILS_H_