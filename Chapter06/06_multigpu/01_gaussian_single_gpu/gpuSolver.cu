#include "gpuSolverFunctions.cu"

void gaussianElimination(unsigned char* A, unsigned char* B, int rowCount, int columnCount)
{
    
    // --- variable used for single gpu code
    int packedColumnCount;

    unsigned int* packedAB;
    unsigned int* packedTransposeAB;

    packedColumnCount = intCeilDiv(columnCount + 1, PACK_SIZE);
 
    // --- Allocate memory for input matrix A on cpu
    long int packedSize = ((long int)sizeof(unsigned int)) * rowCount * packedColumnCount;
    packedAB           = (unsigned int*) malloc(packedSize);
    packedTransposeAB  = (unsigned int*) malloc(packedSize);

    if(packedAB == NULL || packedTransposeAB == NULL)
    {
        printf("Unable to allocate space for packed linear system.\n");
        return;
    }
    packLinearSystem(packedAB, A, B, rowCount, columnCount);
    transposeMatrixCPU(packedTransposeAB, packedAB, rowCount, packedColumnCount);
    gaussianEliminationSingleGPU(packedTransposeAB, rowCount, columnCount);
    transposeMatrixCPU(packedAB, packedTransposeAB, packedColumnCount, rowCount);
    unpackLinearSystem(A, B, packedAB, rowCount, columnCount);
    return;
}

void gaussianEliminationSingleGPU(unsigned int* packedTransposeAB, int rowCount, int columnCount)
{
    int i;

    stSolverState solverState;
    initializeSolver(&solverState, rowCount, columnCount, packedTransposeAB); 
    

    for(i = 0; i < columnCount; i++)
    {
        solverState.columnPack = i / PACK_SIZE;
        solverState.columnBit  = i % PACK_SIZE;
        launchFindPivotRowKernel(&solverState);
        launchExtractPivotRowKernel(&solverState);
        launchRowEliminationKernel(&solverState);
    }
 
    gatherResult(packedTransposeAB, &solverState);
    freeSolver(&solverState);
}


