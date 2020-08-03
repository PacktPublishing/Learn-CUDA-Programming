#pragma once 
#ifndef _GE_SOLVER_H_
#define _GE_SOLVER_H_

    #include <stdlib.h>
    #include <stdio.h>
    #include <string.h>
    #include <vector_types.h>

    #include "config.h"


    //Sanity check on config parameters
    #if (defined(ELEMENT_TYPE_UINT) && defined(ELEMENT_TYPE_UINT4))
        #error Enable only one of the two ELEMENT_TYPE_UINT and ELEMENT_TYPE_UINT4
    #endif

    #if !defined(ELEMENT_TYPE_UINT) && !defined(ELEMENT_TYPE_UINT4)
        #error Enable at least one of the two ELEMENT_TYPE_UINT and ELEMENT_TYPE_UINT4
    #endif

     #if defined(ELEMENT_TYPE_UINT)
        typedef unsigned int elemtype;
        // Number of bits clubbed together to form a pack. All the packed its are processed together.
        #define PACK_SIZE (8*4)
        #define VEC_LEN 1
     #elif defined(ELEMENT_TYPE_UINT4)
        typedef uint4 elemtype;
       // Number of bits clubbed together to form a pack. All the packed its are processed together.
        #define PACK_SIZE (8*16)
        #define VEC_LEN 4
     #endif

    typedef struct  
    {
        /*
            At any point during Gaussian elimination, the results of intermediate computations are
            kept in the elements of this structure. In short this structure serves as context for
            all the solver. This structure holds information that changes on every iteration, such
            as pivot row, multipliers etc. It also holds iteration invariant information such as 
            number of rows and number of columns in the linear system.
            Keeping all the information in a single structure makes it easy to pass it to multiple
            functions.
        */

        // Iteration invariant parameters
        
        int rowCount;          //  Number of rows in linear system
        int columnCount;       //  Number of columns in linear system
        int packedColumnCount; //  Number of packed columns = ceil(columnCount/PACK_SIZE)
        elemtype* h_packedAB; // Packed linear system on host, this is valid only on rank 0 process
	elemtype* h_packedTransposeAB; // Packed transposed linear system on host,valid only on rank 0 
        elemtype* h_myPartOfPackedTransposeAB; // Each process holds its portition of the packed data
	int myPackedColumnCount; // Number of packed columns of the transposed linear system with the current process, since we do only column partitioning, there is no myRowCount
	int myNumChunks;
	int myLastChunkSize;
        int numProcs;
        int myRank;
	elemtype* d_packedTransposeAB; // device pointer to the packed transposed linear system
       
        // Specific to current reduction iteration

        int columnPack;               // Packed column index of current column under reduction = floor(iteration/PACK_SIZE)
        int columnBit;                // Bit index of of current column under reduction = iteration % PACK_SIZE
        unsigned char* isRowReduced;  // isRowReduced[i] is true iff i^th row is already reduced
#ifdef PIVOTPACK
        unsigned int *multipliers;
#else
        unsigned char* multipliers;   // multipliers[i] is 1 if ith row needs to undergo reduction for current iteration
#endif
        int*           pivotRowIndex; // Index of current pivot row
        elemtype*  pivotRow;      // pivot row that should be use for current reduction iteration

    } stSolverState;

    #define CHUNK_SIZE 16 //Data (packed columns) is distributed among the processes using cyclic column partitioning
   
    #ifdef PIVOTPACK
    #define PIVOT_PACK_SIZE (8*4)
    #endif 

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
    
    // Write entries 'vector' to file with name 'filename'. Number of entries equal to 'length' are
    // written to file. The entries are written in plain-text format. Each entry is written on a 
    // new line.
    int writeVectorToFile(const char* filename, unsigned char* vector, int length);
    
    // Prints matrix 'matrixA' to standard output. The arguments 'rowCount' and 'columnCount' are
    // used to specify the dimensions of the matrix. 
    void printMatrix(unsigned char* matrixA, int rowCount, int columnCount);


    // Converts the linear system given by 'A' and 'B' to augmented packed form.
    // Augmented system is obtained by appending one vector 'B' as a column to matrix 'B'.
    // Columns of the augmented system are packed together to form augmented packed linear system.
    // Consecutive entries from each row are grouped together to form packed columns.
    // The augmented packed system is returned in the output argument 'packedAB'.
    void packLinearSystem(elemtype* packedAB, unsigned char* A, unsigned char* B, int rowCount, int columnCount);

    // Unpacks the linear linear system. 
    // This function reverses the action of pack linear system function.
    void unpackLinearSystem(unsigned char* A, unsigned char* B, elemtype* packedAB, int rowCount, int columnCount);
    
    // Transposes matrix 'A'. Output argument 'transposeA' holds the result.
    // Matrix dimension are specified using 'rowcount' and 'columnCount' argument.
    void transposeMatrixCPU(elemtype *transposeA, elemtype *A, int  rowCount, int columnCount);

    // Reads a linear system given by 'matrixA' and 'B' from the file with name 'filename'.
    // The file should be in plain text format.
    // Each line of the file represents one row of the matrix 'A'. Last entry on the row is entry
    // from the vector 'B'.
    void readLinearSystemFromFile(const char* filename, unsigned char* matrixA, unsigned char* B, int rowCount, int columnCount);
    
    // Writes linear system given by 'matrixA' and vector 'B' to the file with name 'filename'.
    // The format of writing is as specified in readLinearSystemFromFile().
    void writeLinearSystemFromFile(const char* filename, unsigned char* matrixA, unsigned char* B, int rowCount, int columnCount);

    // This function is called by rank 0. It internally calls packLinearSystem() and transposeMatrixCPU()
    int prepareDataforGaussianElimination(unsigned char* A, unsigned char* B, int rowCount, int columnCount, stSolverState *solverState);

    // Top level wrapper for Gaussian elimination.
    int gaussianElimination(stSolverState *solverState, int myRank, int nodeLocalRank, int numProcs);

    // Gaussian elimination on GPU
    void gaussianEliminationOnGPU(stSolverState *solverState, int numProcs, int myRank, int myNodeLocalRank);

    void distributeInputs(stSolverState *solverState, int myRank, int nodeLocalRank, int numProcs); // Rank 0 distributes input data to all procs
    void gatherResult(stSolverState *solverState, int myRank, int nodeLocalRank, int numProcs); // Rank 0 gathers result from all procs

    // This function broadcasts the pivotRow index and multipliers
    void bcastPivotRow(int pivotProc, int myRank, int numProcs, stSolverState *solverState, void **commReq, void **commStatus);
    void waitForPivotBcast(int pivotProc, void *req, void *status, int numProcs, int myRank);
    
    // This function is called by rank 0. It internally calls transposeMatrixCPU() and unpackLinearSystem()
    void prepareResultFromGaussianElimination(unsigned char* A, unsigned char* B, int rowCount, int columnCount, stSolverState *solverState);

    // Performs backsubstitution operation.
    // Argument 'matrixA' is assumed to be upper triangular matrix.
    // Argument 'B' is a vector with 'number_of_equations' entries.
    // Output argument 'x' is solution vector
    int backsubstitution(unsigned char* matrixA, unsigned char* B, unsigned char* x, int number_of_equations, int number_of_variables);
    
    // Allocates GPU memory for fields of 'solverState' and also initializes them
    void initializeSolver(stSolverState* solverState, int rowCount, int columnCount, int rank, int nodeLocalRank, int numProcs);

    // Free GPU memory allocated for fields of 'solverState'
    void freeSolver(stSolverState* solverState);

    void launchFindPivotRowKernel(stSolverState *solverState);
    void launchExtractPivotRowKernel(stSolverState *solverState);
    void launchRowEliminationKernel(stSolverState *solverState);

    // Utitlities for operating bits of elemtype
    // Make all bits zero
    void resetElement(elemtype* e);

    // Make 'bitIndex' bit 1
    void setElementBit(elemtype* e, int bitIndex);

    // Get 'bitIndex' bit
    unsigned char getElementBit(elemtype* e, int bitIndex);

    #ifdef __NVCC__
        // Extract bit-field. Bits of 'x' are numbered 0-31.
        // Bit-field with 'numBits' number of bits starting from 'bit' bit is returned
        // for example 'bit' = 0 and 'numBits' = 5, returns five least significant bits of 'x' 
        __device__ unsigned int bfe(unsigned int x, unsigned int bit, unsigned int numBits);

        // Same as bfe() but for elemtype datatype
         __device__ unsigned int bfew(elemtype* e, int bitIndex);
        
        // CUDA error reporting
        void _report(cudaError_t result, char const *const func, const char *const file, int const line);
        
        // Block level reduction operation to find max.
        // 'sData' is array with 'blockSize' entries. 
        // 'tid' is local thread index.
        // After execution sData[0] contains the largest entry from 'sData'.
        __device__ void findMax(int* sData, const unsigned int tid, const unsigned int blockSize);
        
        __global__ void findPivotRowAndMultipliers(stSolverState solverState, elemtype* const __restrict__ d_packedTransposeAB);
        __global__ void extractPivotRow(stSolverState solverState, elemtype* d_packedTransposeAB);
        __global__ void rowElimination(elemtype* d_packedTransposeAB, stSolverState solverState);
    #endif

#endif
