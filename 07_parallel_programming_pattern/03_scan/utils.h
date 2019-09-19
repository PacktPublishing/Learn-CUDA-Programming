#ifndef _UTILS_H_
#define _UTILS_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdarg.h>

// generate input data
void generate_data(float *ptr, int length)
{
    // fill the buffer with random generated unsigned integers
    for (int i = 0; i < length; i++)
        ptr[i] = (rand() - RAND_MAX/2) / (float)RAND_MAX;
}

bool validation(float *a, float *b, int length)
{
    float epsilon = 0.000001;
    bool result = true;
    for (int i = 0; i < length; i++) {
        if (abs(a[i] - b[i]) >= epsilon) {
            result = false;
            printf("result mismatch on %d th item. (%f) \n", i, abs(a[i] - b[i]));
        }
    }
    return result;
}

void print_val(float *h_list, int length, ...)
{
    va_list argptr;
    va_start(argptr, length);

    printf("%s\t", va_arg(argptr, char *));
    for (int i = 0; i < length; i++)
        printf("%7.4f\t", h_list[i]);
    printf("\n");
}

#endif  // _UTILS_H_