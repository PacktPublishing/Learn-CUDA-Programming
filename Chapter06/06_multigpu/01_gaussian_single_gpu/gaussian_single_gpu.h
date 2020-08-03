#ifndef _GAUSSIAN_SOLVER_H_
#define _GAUSSIAN_SOLVER_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "config.h"

typedef struct  
{
	/*
	   At any point during Gaussian elimination, the results of intermediate computations are kept in the elements of this structure. 
	   */
	int rowCount;          //  Number of rows in linear system
	int columnCount;       //  Number of columns in linear system
	int packedColumnCount; //  Number of packed columns = ceil(columnCount/PACK_SIZE)
	unsigned int *d_packedTransposeAB; // Device copy of augmented, packed and transposed linear system 

	int columnPack;               // Packed column index of current column under reduction = floor(iteration/PACK_SIZE)
	int columnBit;                // Bit index of current column under reduction = iteration % PACK_SIZE
	unsigned char* isRowReduced;  // isRowReduced[i] is true iff i^th row is already reduced
	unsigned char* multipliers;   // multipliers[i] is 1 if ith row needs to undergo reduction for current iteration
	int*           pivotRowIndex; // Index of current pivot row
	unsigned int*  pivotRow;      // pivot row that should be use for current reduction iteration

} stSolverState;

// Number of bits clubbed together to form a pack. All the packed its are processed together.
#define PACK_SIZE (8*sizeof(unsigned int))

// Ceil of numerator divided by denominator.
#define intCeilDiv(numerator, denominator) (((numerator) + (denominator) - 1)/ (denominator))

// CUDA error reporting.
#define checkCudaErrors(val) _report( (val), #val, __FILE__, __LINE__ ) 


// Generates a linear system with pseudo-random entries.
// 'A' is generated as a matrix with 'rowCount' rows and 'columnCount' columns.
// 'x' is generated as a vector with 'columnCount' entries.
// vector 'B' has 'rowCount' entries and is computed as B = A*x.
// Value of percent can be 0 to 100. The generated matrix 'A' has 'percent' % non-zero entries.
void generateLinearSystem(unsigned char *A, unsigned char* B, unsigned char* x, int rowCount, int columnCount, int percent);

void writeVectorToFile(const char* filename, unsigned char* vector, int length);

// Converts the linear system given by 'A' and 'B' to augmented packed form.
// Augmented system is obtained by appending one vector 'B' as a column to matrix 'B'.
// Columns of the augmented system are packed together to form augmented packed linear system.
// Consecutive entries from each row are grouped together to form packed columns.
// The augmented packed system is returned in the output argument 'packedAB'.
void packLinearSystem(unsigned int* packedAB, unsigned char* A, unsigned char* B, int rowCount, int columnCount);

// Unpacks the linear linear system. 
// This function reverses the action of pack linear system function.
void unpackLinearSystem(unsigned char* A, unsigned char* B, unsigned int* packedAB, int rowCount, int columnCount);

// Transposes matrix 'A'. Output argument 'transposeA' holds the result.
// Matrix dimension are specified using 'rowcount' and 'columnCount' argument.
void transposeMatrixCPU(unsigned int *transposeA, unsigned int *A, int  rowCount, int columnCount);

void writeLinearSystemFromFile(const char* filename, unsigned char* matrixA, unsigned char* B, int rowCount, int columnCount);

// Top level wrapper for Gaussian elimination.
void gaussianElimination(unsigned char *A, unsigned char* B, int rowCount, int columnCount);

int backsubstitution(unsigned char* matrixA, unsigned char* B, unsigned char* x, int number_of_equations, int number_of_variables);

// Single GPU Gaussian elimination wrapper function.
void gaussianEliminationSingleGPU(unsigned int* packedTransposeAB, int rowCount, int columnCount);

// Allocates GPU memory for fields of 'solverState' and also initializes them
void initializeSolver(stSolverState* solverState, int rowCount, int columnCount, unsigned int *h_packedTransposeAB);

// Free GPU memory allocated for fields of 'solverState'
void freeSolver(stSolverState* solverState);

void launchFindPivotRowKernel(stSolverState *solverState); // Finding the pivot row 
void launchExtractPivotRowKernel(stSolverState *solverState); // extracting the pivot row 
void launchRowEliminationKernel(stSolverState *solverState); // Row elimination
void gatherResult(unsigned int *h_packedTransposeAB, stSolverState *solverState); // Gathers result of the linear solver into h_packedTransposeAB

// Extract bit-field. Bits of 'x' are numbered 0-31.
// Bit-field with 'numBits' number of bits starting from 'bit' bit is returned
// for example 'bit' = 0 and 'numBits' = 5, returns five least significant bits of 'x' 
__device__ unsigned int bfe(unsigned int x, unsigned int bit, unsigned int numBits);

// CUDA error reporting
void _report(cudaError_t result, char const *const func, const char *const file, int const line);

// Block level reduction operation to find max.
// 'sData' is array with 'blockSize' entries. 
// 'tid' is local thread index.
// After execution sData[0] contains the largest entry from 'sData'.
__device__ void findMax(int* sData, const unsigned int tid, const unsigned int blockSize);

__global__ void findPivotRowAndMultipliers(stSolverState solverState, unsigned int* d_packedTransposeAB);
__global__ void extractPivotRow(stSolverState solverState, unsigned int* d_packedTransposeAB);
__global__ void rowElimination(unsigned int* d_packedTransposeAB, stSolverState solverState);
#endif
