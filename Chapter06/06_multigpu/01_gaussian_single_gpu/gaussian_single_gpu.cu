#define LINK_HEADER_LIBS
#include "gaussian_single_gpu.h"
#include "gpuSolver.cu"
#include "linearSystemOps.cu"

int main(int argc, char** argv)
{

    unsigned long int rowCount, columnCount;
    unsigned long int percent;

    // --- variable used for single threaded cpu code
    unsigned char* A;
    unsigned char* B;
    unsigned char* x;

    rowCount = ROWS;
    columnCount = COLS;
    printf("Number of arguments = %lu\n", argc);
    if(argc == 3) {
    	rowCount = atoi(argv[1]);
    	columnCount = atoi(argv[2]);
    }
    percent = PERCENTAGE;

    printf("Running solver with following parameters:\n");
    printf("Number of rows = %lu\n", rowCount);
    printf("Number of columns = %lu\n", columnCount);
    printf("Percentage density = %lu\n", percent);


    // --- Allocate memory for input matrix A and B on cpu
    A = (unsigned char*)malloc(sizeof(unsigned char) * rowCount * columnCount);
    B = (unsigned char*)malloc(sizeof(unsigned char) * rowCount);
    x = (unsigned char*)malloc(sizeof(unsigned char) * columnCount);    

    if(A == NULL || B == NULL || x == NULL)
    {
        printf("Unable to allocate space for linear system\n");
        exit(-1);
    }

    // --- Initialise the input matrix
    generateLinearSystem(A, B, x, rowCount, columnCount, percent);
    writeVectorToFile(REFERENCE_SOLUTION, x, columnCount);
    
    gaussianElimination(A, B, rowCount, columnCount);
    
    backsubstitution(A, B, x, rowCount, columnCount);
    writeVectorToFile(COMPUTED_SOLUTION, x, columnCount);
    
    free(x);
    free(A);
    free(B);

    return 0;
}

