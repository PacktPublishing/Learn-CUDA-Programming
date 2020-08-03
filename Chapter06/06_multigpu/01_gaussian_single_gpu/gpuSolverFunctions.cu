void _report(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if(result)
    {
        fprintf(stderr, "CUDA error at %s:%d code = %d (%s) \"%s\" \n",
                file, line, result, cudaGetErrorString(result), func);
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(-1);
    }
}

__device__ unsigned int bfe(unsigned int x, unsigned int bit, unsigned int numBits)
{
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
    return ret;
}

void initializeSolver(stSolverState* solverState, int rowCount, int columnCount, unsigned int *h_packedTransposeAB)
{
    int packedColumnCount = intCeilDiv(columnCount+1, PACK_SIZE);
    long int packedSize   = sizeof(unsigned int)*rowCount*packedColumnCount;

    solverState->rowCount = rowCount;
    solverState->columnCount = columnCount;
    solverState->packedColumnCount = packedColumnCount;

    checkCudaErrors(cudaMalloc((void**)&(solverState->d_packedTransposeAB), packedSize));
    checkCudaErrors(cudaMemcpy(solverState->d_packedTransposeAB, h_packedTransposeAB,  packedSize, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&(solverState->pivotRowIndex), sizeof(int)));

    checkCudaErrors(cudaMalloc((void**)&(solverState->multipliers), sizeof(unsigned char) * rowCount));
    checkCudaErrors(cudaMalloc((void**)&(solverState->isRowReduced), sizeof(unsigned char) * rowCount));
    checkCudaErrors(cudaMalloc((void**)&(solverState->pivotRow), packedColumnCount * sizeof(unsigned int)));

    checkCudaErrors(cudaMemset(solverState->isRowReduced, 0, sizeof(unsigned char) * rowCount));

}

void freeSolver(stSolverState* solverState)
{
    checkCudaErrors(cudaFree(solverState->d_packedTransposeAB));
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
    extractPivotRowBlocks.x = intCeilDiv(solverState->packedColumnCount, extractPivotRowThreads.x);
    extractPivotRowBlocks.y = 1;
    extractPivotRowBlocks.z = 1;
    
    extractPivotRow<<<extractPivotRowBlocks, extractPivotRowThreads>>>(*solverState, solverState->d_packedTransposeAB);
}

void launchRowEliminationKernel(stSolverState *solverState)
{
    dim3 rowEliminationThreads(1024, 1, 1);
    dim3 rowEliminationBlocks;
    rowEliminationBlocks.x = intCeilDiv(solverState->rowCount, (4*rowEliminationThreads.x));
    rowEliminationBlocks.y = intCeilDiv(solverState->packedColumnCount, rowEliminationThreads.y);
    rowEliminationBlocks.z = 1;

    rowElimination<<<rowEliminationBlocks, rowEliminationThreads>>>(solverState->d_packedTransposeAB, *solverState);
}

void gatherResult(unsigned int *h_packedTransposeAB, stSolverState *solverState)
{
    long int packedSize   = sizeof(unsigned int)*solverState->rowCount*solverState->packedColumnCount;
    checkCudaErrors(cudaMemcpy(h_packedTransposeAB, solverState->d_packedTransposeAB, packedSize, cudaMemcpyDeviceToHost));
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

__global__ void findPivotRowAndMultipliers(stSolverState solverState, unsigned int* const __restrict__ d_packedTransposeAB)
{
    __shared__ int  eligibleRow[128];
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    bool isEligible = false;

    if(threadIdx.x == 0 && blockIdx.x == 0) 
        *solverState.pivotRowIndex = -1;

    if(row < solverState.rowCount)
    {
        long int index = row + ((long int)solverState.columnPack)*solverState.rowCount;
        isEligible =  (
                        bfe(d_packedTransposeAB[index], solverState.columnBit, 1) && 
                        (solverState.isRowReduced[row] == 0)
                      ) ? true:false;

        eligibleRow[threadIdx.x]  = (isEligible == true) ? row : -1;
        solverState.multipliers[row] = isEligible;
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

__global__ void extractPivotRow(stSolverState solverState, unsigned int* d_packedTransposeAB)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    long int index = ((long int)col)*solverState.rowCount + (*solverState.pivotRowIndex);
    if(col < solverState.packedColumnCount)
    {
        solverState.pivotRow[col] = d_packedTransposeAB[index];
    }

    if(col == 0)
    {
        //printf("pivotRowIndex = %d rowCount = %d packedColumnCount = %d\n", *solverState.pivotRowIndex, solverState.rowCount, solverState.packedColumnCount);
        solverState.isRowReduced[*solverState.pivotRowIndex] = 1;
        solverState.multipliers[*solverState.pivotRowIndex]  = 0;
    }
}

__global__ void rowElimination(unsigned int* d_packedTransposeAB, stSolverState solverState)
{
    int row, col, r;
    row = blockIdx.x * blockDim.x + threadIdx.x;
    col = blockIdx.y * blockDim.y;
    long int reductionElement;

    if (col >= solverState.packedColumnCount ||
        col < solverState.columnPack)
        return;
   // int laneID = threadIdx.x & 0x1f;
    //unsigned int pivotValue;

    //if (laneID == 0) pivotValue = solverState.pivotRow[col];
    //pivotValue = __shfl(pivotValue, 0);
    reductionElement = (long int) (col)*solverState.rowCount;
    r = row;
#pragma unroll 4 
    for (int r = row; r < solverState.rowCount; r+=blockDim.x*gridDim.x)
    {
        if(solverState.multipliers[r])
        {
        #if (__CUDA_ARCH__ >= 350)
           d_packedTransposeAB[reductionElement + r] ^= __ldg(&solverState.pivotRow[col]);
        #else
            d_packedTransposeAB[reductionElement + r] ^= solverState.pivotRow[col];
       //     d_packedTransposeAB[reductionElement] ^= pivotValue;
       #endif
        }
    }
}



