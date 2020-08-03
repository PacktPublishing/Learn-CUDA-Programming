#include "gaussian_multi_gpu_rdma.h"

void gaussianEliminationOnGPU(stSolverState *solverState, int numProcs, int myRank, int myNodeLocalRank)
{
    double bcastTime = 0.0;
    long int packedSize;
    int i; 
    void **req, **status;
#ifdef PIVOTPACK
    int packedRowCount;
#endif

    req = (void **)malloc(sizeof(void *));
    status = (void **)malloc(sizeof(void *));
    dim3 findPivotThreads(128, 1, 1); 
    dim3 findPivotBlocks(intCeilDiv(solverState->rowCount, 128), 1, 1); 

    dim3 extractPivotRowThreads(1024, 1, 1);
    dim3 extractPivotRowBlocks;

    extractPivotRowBlocks.x = intCeilDiv(solverState->myPackedColumnCount, extractPivotRowThreads.x);
    extractPivotRowBlocks.y = 1;
    extractPivotRowBlocks.z = 1;

    dim3 rowEliminationThreads(512, 1, 1);
    dim3 rowEliminationBlocks;

    rowEliminationBlocks.x = intCeilDiv(solverState->rowCount, (2*rowEliminationThreads.x));
    rowEliminationBlocks.y = intCeilDiv(solverState->myPackedColumnCount, rowEliminationThreads.y);
    rowEliminationBlocks.z = 1;
   

    checkCudaErrors(cudaSetDevice(myNodeLocalRank));
    packedSize = (long int)(solverState->rowCount * solverState->myNumChunks * CHUNK_SIZE * sizeof(elemtype));
    checkCudaErrors(cudaMalloc((void**)&(solverState->d_packedTransposeAB), packedSize));
    checkCudaErrors(cudaMemset(solverState->d_packedTransposeAB, 0, packedSize));
    checkCudaErrors(cudaMemcpy(solverState->d_packedTransposeAB, solverState->h_myPartOfPackedTransposeAB, packedSize, cudaMemcpyHostToDevice));


#ifdef PIVOTPACK
    packedRowCount = intCeilDiv(solverState->rowCount, PIVOT_PACK_SIZE);
#endif
    for(i = 0; i < solverState->columnCount; i++)
    { 
        int pivotProc, chunkNumber; //Process that holds data corresponding to ith column
       
        chunkNumber = (i / PACK_SIZE)/CHUNK_SIZE;
        pivotProc = (chunkNumber % numProcs); 
        //printf("Pivot Proc %d\n", pivotProc);
#ifdef PIVOTPACK
        checkCudaErrors(cudaMemset(solverState->multipliers, 0, sizeof(unsigned int) * packedRowCount));
#endif
        if (myRank == pivotProc) // Only the process that holds i, does the following
        {
            int localChunkIndex;

            localChunkIndex = (chunkNumber/numProcs);    
            solverState->columnPack  = (localChunkIndex * CHUNK_SIZE) + ((i /PACK_SIZE) % CHUNK_SIZE);
            solverState->columnBit   = i % PACK_SIZE;
            //printf("columnpack %d, column bit %d\n", solverState->columnPack, solverState->columnBit);
            findPivotRowAndMultipliers<<<findPivotBlocks, findPivotThreads>>>((*solverState), solverState->d_packedTransposeAB);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        /* Rank pivotProc needs to broadcast the pivotRowIndex and multipliers */
        bcastPivotRow(pivotProc, myRank, numProcs, solverState, req, status);
        solverState->columnPack  = (i /PACK_SIZE);
        solverState->columnBit   = i % PACK_SIZE;
        extractPivotRow<<<extractPivotRowBlocks, extractPivotRowThreads>>>(*solverState, solverState->d_packedTransposeAB);
        rowElimination<<<rowEliminationBlocks, rowEliminationThreads>>>(solverState->d_packedTransposeAB, *solverState);
        waitForPivotBcast(pivotProc, *req, *status, numProcs, myRank);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    free(req);
    free(status);

    checkCudaErrors(cudaMemcpy(solverState->h_myPartOfPackedTransposeAB, solverState->d_packedTransposeAB, packedSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    return; 
}
