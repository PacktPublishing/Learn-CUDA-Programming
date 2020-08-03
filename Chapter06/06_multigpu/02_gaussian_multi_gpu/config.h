#ifndef CONFIG_H
#define CONFIG_H

#define INPUT_TYPE RANDOM

// Linear system parameters
#define ROWS 300           // Number of rows in the system.
#define COLS 256           // Number of columns in the system
#define PERCENTAGE 50       // Density of coefficient matrix

#define REFERENCE_SOLUTION "original-matrix"
#define COMPUTED_SOLUTION  "computed-solution"

#define PACK_SIZE (8*sizeof(unsigned int))
// Ceil of numerator divided by denominator.
#define intCeilDiv(numerator, denominator) (((numerator) + (denominator) - 1)/ (denominator))

// How many GPUs should be used by solver. Effective only with MULTI_GPU
#define NUMBER_OF_GPU 2

#endif
