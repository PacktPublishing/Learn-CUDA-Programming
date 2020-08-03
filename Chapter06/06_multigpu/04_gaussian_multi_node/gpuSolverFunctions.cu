#include "gaussian_multi_gpu_rdma.h"


void initializeSolver(stSolverState* solverState, int rowCount, int columnCount, int myRank, int myNodeLocalRank, int numProcs)
{
    int packedColumnCount = intCeilDiv(columnCount+1, PACK_SIZE);
#ifdef PIVOTPACK
    int packedRowCount = intCeilDiv(rowCount, PIVOT_PACK_SIZE);
#endif

    memset(solverState, 0, sizeof(stSolverState));
    solverState->rowCount = rowCount;
    solverState->columnCount = columnCount;
    solverState->packedColumnCount = packedColumnCount;

    solverState->myRank = myRank;
    solverState->numProcs = numProcs;

    checkCudaErrors(cudaSetDevice(myNodeLocalRank));

    checkCudaErrors(cudaMalloc((void**)&(solverState->pivotRowIndex), sizeof(int)));

#ifdef PIVOTPACK
    checkCudaErrors(cudaMalloc((void**)&(solverState->multipliers), sizeof(unsigned int) * packedRowCount));
    checkCudaErrors(cudaMemset(solverState->multipliers, 0, sizeof(unsigned int) * packedRowCount));
#else
    checkCudaErrors(cudaMalloc((void**)&(solverState->multipliers), sizeof(unsigned char) * rowCount));
#endif
    checkCudaErrors(cudaMalloc((void**)&(solverState->isRowReduced), sizeof(unsigned char) * rowCount));
    checkCudaErrors(cudaMalloc((void**)&(solverState->pivotRow), packedColumnCount * sizeof(elemtype)));

    checkCudaErrors(cudaMemset(solverState->isRowReduced, 0, sizeof(unsigned char) * rowCount));
    return;
}

void freeSolver(stSolverState* solverState)
{
    checkCudaErrors(cudaFree(solverState->pivotRowIndex));
    checkCudaErrors(cudaFree(solverState->multipliers));
    checkCudaErrors(cudaFree(solverState->isRowReduced));
    checkCudaErrors(cudaFree(solverState->pivotRow));
}

void launchFindPivotRowKernel(stSolverState *solverState)
{
    dim3 findPivotThreads(128, 1, 1);
    dim3 findPivotBlocks(intCeilDiv(solverState->rowCount, 128), 1, 1);

    findPivotRowAndMultipliers<<<findPivotBlocks, findPivotThreads>>>(*solverState, solverState->d_packedTransposeAB);
}

void launchExtractPivotRowKernel(stSolverState *solverState)
{
    dim3 extractPivotRowThreads(1024, 1, 1);
    dim3 extractPivotRowBlocks;
    extractPivotRowBlocks.x = intCeilDiv(solverState->myPackedColumnCount, extractPivotRowThreads.x);
    extractPivotRowBlocks.y = 1;
    extractPivotRowBlocks.z = 1;

    extractPivotRow<<<extractPivotRowBlocks, extractPivotRowThreads>>>(*solverState, solverState->d_packedTransposeAB);
}

void launchRowEliminationKernel(stSolverState *solverState)
{
    dim3 rowEliminationThreads(512, 1, 1);
    dim3 rowEliminationBlocks;
    rowEliminationBlocks.x = intCeilDiv(solverState->rowCount, (2*rowEliminationThreads.x));
    rowEliminationBlocks.y = intCeilDiv(solverState->myPackedColumnCount, rowEliminationThreads.y);
    rowEliminationBlocks.z = 1;

    rowElimination<<<rowEliminationBlocks, rowEliminationThreads>>>(solverState->d_packedTransposeAB, *solverState);
}


// Solver Kernels

__device__ void findMax(int* sData, const unsigned int tid, const unsigned int blockSize)
{
    unsigned int stride = blockSize/2;

    for(; stride >= 1; stride /= 2)
    {
       if (tid < stride)
        {
            if(sData[tid] < sData[tid+stride])
                sData[tid] = sData[tid+stride];
        }
         __syncthreads(); 
    }
}

__global__ void findPivotRowAndMultipliers(stSolverState solverState, elemtype* const __restrict__ d_packedTransposeAB)
{
    __shared__ int  eligibleRow[128];
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int isEligible = 0;
#ifdef PIVOTPACK
    int rowPack, rowBit;
#endif

    if(threadIdx.x == 0 && blockIdx.x == 0) 
        *solverState.pivotRowIndex = -1;

#ifdef PIVOTPACK
     rowPack = row/PIVOT_PACK_SIZE;
     rowBit = row % PIVOT_PACK_SIZE;
#endif

    if(row < solverState.rowCount)
    {
        long int index = row + ((long int)solverState.columnPack)*solverState.rowCount;
        isEligible =  (
                        bfew(&d_packedTransposeAB[index], solverState.columnBit) && 
                        (solverState.isRowReduced[row] == 0)
                      ) ? 1:0;

        eligibleRow[threadIdx.x]  = (isEligible == 1) ? row : -1;
#ifdef PIVOTPACK
        atomicOr(&solverState.multipliers[rowPack], (isEligible << rowBit));
#else
        solverState.multipliers[row] = isEligible;
#endif
    }
    else
    {
        eligibleRow[threadIdx.x] = -1;
    }

    __syncthreads();

    findMax(eligibleRow, threadIdx.x, blockDim.x);

    if(threadIdx.x == 0)
    {
        atomicMax(solverState.pivotRowIndex, eligibleRow[0]);
    }

}

__global__ void extractPivotRow(stSolverState solverState, elemtype* d_packedTransposeAB)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    long int index = ((long int)col)*solverState.rowCount + (*solverState.pivotRowIndex);
#ifdef PIVOTPACK
    int rowPack = (*solverState.pivotRowIndex)/PIVOT_PACK_SIZE;
    int rowBit = (*solverState.pivotRowIndex) % PIVOT_PACK_SIZE;
#endif
    if(col < solverState.myPackedColumnCount)
    {
        solverState.pivotRow[col] = d_packedTransposeAB[index];
    }

    if(col == 0)
    {
        solverState.isRowReduced[*solverState.pivotRowIndex] = 1;
#ifdef PIVOTPACK
        atomicAnd(&solverState.multipliers[rowPack], ~(1 << rowBit));
#else
        solverState.multipliers[*solverState.pivotRowIndex]  = 0;
#endif
    }
}

__global__ void rowElimination(elemtype* d_packedTransposeAB, stSolverState solverState)
{
    int row, col, gChunkNo, gCol;
    row = blockIdx.x * blockDim.x + threadIdx.x;
    col = blockIdx.y * blockDim.y;
    long int reductionElement;
   
    gChunkNo = ((col / CHUNK_SIZE) * solverState.numProcs) + solverState.myRank;
    gCol = (gChunkNo * CHUNK_SIZE) + (col % CHUNK_SIZE);

    if(
        row >= solverState.rowCount ||
        col >= solverState.myPackedColumnCount ||
        gCol <  solverState.columnPack)
        return;
    reductionElement = ((long int)col)*solverState.rowCount;
    elemtype pivotElem;

    #if (__CUDA_ARCH__ >= 350)
       pivotElem = __ldg(&solverState.pivotRow[col]);
    #else
       pivotElem = solverState.pivotRow[col];
    #endif


#pragma unroll 4
    for (int r = row; r < solverState.rowCount; r+=blockDim.x*gridDim.x)
    {
        unsigned char multiplierVal;
        
#ifdef PIVOTPACK
        int rowPack = r/PIVOT_PACK_SIZE;
        int rowBit = r % PIVOT_PACK_SIZE;
        multiplierVal = bfe(solverState.multipliers[rowPack], rowBit, 1);
#else
        multiplierVal = solverState.multipliers[r];
#endif
        if (multiplierVal)
        {
             elemtype matElem = d_packedTransposeAB[reductionElement + r];
        #if (PACK_SIZE == 32)
            matElem ^= pivotElem;
        #elif (PACK_SIZE == 128)
            matElem.x ^= pivotElem.x;
            matElem.y ^= pivotElem.y;
            matElem.z ^= pivotElem.z;
            matElem.w ^= pivotElem.w;
        #else
            #error "Unsupported PACK_SIZE detected"
        #endif
        d_packedTransposeAB[reductionElement + r] = matElem;

        }
    }
}
