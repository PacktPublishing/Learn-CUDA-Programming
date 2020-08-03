#define LINK_HEADER_LIBS
#include <stdio.h>
#include "gaussian_multi_gpu_rdma.h"
#include "mpiUtils.h"

int main(int argc, char* argv[])
{
    int numProcs, myRank;
    unsigned long int rowCount, columnCount;
    MPI_Comm nodeLocalComm;
    int nodeLocalRank;

    stSolverState solverState;

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numProcs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));

    rowCount = ROWS;
    columnCount = COLS;
    if (argc == 3)
    {
       rowCount = atoi(argv[1]);
       columnCount = atoi(argv[2]);
    }

#ifdef GE_DEBUG
    if (myRank == 0)
        printf("Starting GE Solver with %d MPI processes\n", numProcs);
#endif
    /* There is one process per GPU, hence #processes per node is #gpus per node. We need to find node local MPI rank and that will be the gpu id.
    One way to get the node local rank in openmpi is the environment variable OMPI_COMM_WORLD_LOCAL_RANK. But this is not a portable solution
    So, we split the communicator into sub-groups based on the node, and get the local rank */

    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeLocalComm));
    MPI_CHECK(MPI_Comm_rank(nodeLocalComm, &nodeLocalRank));

    initializeSolver(&solverState, rowCount, columnCount, myRank, nodeLocalRank, numProcs);

    if (myRank == 0)
    {
        unsigned long int percent;
        unsigned char* A = NULL; 
        unsigned char* B = NULL;
        unsigned char* x = NULL;

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
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

    // --- Initialise the input matrix


        #if (INPUT_TYPE == RANDOM)
        generateLinearSystem(A, B, x, rowCount, columnCount, percent);
        if (writeVectorToFile(REFERENCE_SOLUTION, x, columnCount) != 0)
            MPI_Abort(MPI_COMM_WORLD, -1);
        #elif (INPUT_TYPE == FILE)
            readLinearSystemFromFile(INPUT_FILENAME, A, B, rowCount, columnCount);
        #endif

        if (prepareDataforGaussianElimination(A, B, rowCount, columnCount, &solverState) != 0)
             MPI_Abort(MPI_COMM_WORLD, -1);

        free(x);
        free(A);
        free(B);
    }
    if (gaussianElimination(&solverState, myRank, nodeLocalRank, numProcs) != 0)
        MPI_Abort(MPI_COMM_WORLD, -1);
    // Rank 0 prepares result

    if (myRank == 0)
    {
        unsigned char* A = NULL; 
        unsigned char* B = NULL;
        unsigned char* x = NULL;

        // --- Allocate memory for result matrix A and B and x on cpu
        A = (unsigned char*)malloc(sizeof(unsigned char) * rowCount * columnCount);
        B = (unsigned char*)malloc(sizeof(unsigned char) * rowCount);
        x = (unsigned char*)malloc(sizeof(unsigned char) * columnCount);

        if(A == NULL || B == NULL || x == NULL)
        {
            printf("Unable to allocate space for linear system\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        prepareResultFromGaussianElimination(A, B, rowCount, columnCount, &solverState);    
        
        backsubstitution(A, B, x, rowCount, columnCount);

        fflush(stdout);
        writeVectorToFile(COMPUTED_SOLUTION, x, columnCount);

        free(x);
        free(A);
        free(B);
    }
    freeSolver(&solverState);
    MPI_CHECK(MPI_Finalize());
    
    return 0;
}
    
// Please note that this function is called only by Rank 0

int prepareDataforGaussianElimination(unsigned char* A, unsigned char* B, int rowCount, int columnCount, stSolverState *solverState)
{

    int packedColumnCount;

    packedColumnCount = intCeilDiv(columnCount + 1, PACK_SIZE);
    long int packedSize = ((long int)sizeof(elemtype)) * rowCount * packedColumnCount;
    solverState->h_packedAB           = (elemtype*) malloc(packedSize);
    solverState->h_packedTransposeAB  = (elemtype*) malloc(packedSize);

    if(solverState->h_packedAB == NULL || solverState->h_packedTransposeAB == NULL)
    {
        printf("Unable to allocate space for packed linear system.\n");
        return -1;
    }

    packLinearSystem(solverState->h_packedAB, A, B, rowCount, columnCount);
    transposeMatrixCPU(solverState->h_packedTransposeAB, solverState->h_packedAB, rowCount, packedColumnCount);
    return 0;
}


int gaussianElimination(stSolverState *solverState, int myRank, int nodeLocalRank, int numProcs)
{

#ifdef GE_DEBUG
    printf("Rank %d: Entering gaussianElimination()\n", myRank);
#endif

    // Columns are partitioned among the processes
    distributeInputs(solverState, myRank, nodeLocalRank, numProcs);

    gaussianEliminationOnGPU(solverState, numProcs, myRank, nodeLocalRank);

    // Rank 0 recieves the solution partitions from the other procs
    gatherResult(solverState, myRank, nodeLocalRank, numProcs);

    //Cleanup
    free(solverState->h_myPartOfPackedTransposeAB);
#ifdef GE_DEBUG
    printf("Rank %d: Exiting gaussianElimination()\n", myRank);
#endif
    return 0;
}     
    
void distributeInputs(stSolverState *solverState, int myRank, int nodeLocalRank, int numProcs)
{
    int numChunks, numChunksPerProc, lastChunkSize, lastChunkProc, lastChunkOffset = 0;
    long int packedSize, packedCount, packedDataCountPerProc, packedDataSizePerProc;

    numChunks = intCeilDiv(solverState->packedColumnCount, CHUNK_SIZE);
    numChunksPerProc = intCeilDiv(numChunks, numProcs);
    packedDataCountPerProc = (long int) (solverState->rowCount * numChunksPerProc * CHUNK_SIZE);
    packedDataSizePerProc = (long int)(packedDataCountPerProc * sizeof(elemtype));
    //printf("NumChunks %d, nuChunksPerProc %d, packedDataCountPerProc %d, packedDataSizePerProc %d\n", numChunks, numChunksPerProc, packedDataCountPerProc, packedDataSizePerProc);
    solverState->h_myPartOfPackedTransposeAB = (elemtype*)malloc(packedDataSizePerProc);
    if (solverState->h_myPartOfPackedTransposeAB == NULL)
    {
        printf("Unable to allocate space for packed transpose on rank %d process\n", myRank);
        return;
    }
    memset(solverState->h_myPartOfPackedTransposeAB, 0, packedDataSizePerProc);
    lastChunkSize = solverState->packedColumnCount - ((numChunks - 1) * CHUNK_SIZE); 
    packedCount = solverState->rowCount * CHUNK_SIZE;
    packedSize = packedCount * sizeof(elemtype);
    solverState->myNumChunks = 0;
    solverState->myLastChunkSize = CHUNK_SIZE;

   // Rank 0 distributes the partitions of packedTransposeAB to other procs
    if (myRank == 0)
    {
        int c = 0, index = 0;

        lastChunkProc = 0;       
        while (c < (numChunks - 1))
        {
            int r; 

            r = c % numProcs;
        #ifdef GE_DEBUG
            printf("Rank %d: Sending packedTransposeAB to rank %d\n", myRank, r);
        #endif
            if (r == 0)
            {
                memcpy((&solverState->h_myPartOfPackedTransposeAB[index * packedCount]), (&solverState->h_packedTransposeAB[c * packedCount]), packedSize);
                solverState->myNumChunks++;
                index++;
            }
            else
            {  
                MPI_CHECK(MPI_Send((void *)(solverState->h_packedTransposeAB + (c * packedCount)), packedCount*VEC_LEN, MPI_UNSIGNED, r, 0, MPI_COMM_WORLD));
            }
            c++;
            if (c == (numChunks - 1)) 
            {
                lastChunkProc = (r + 1) % numProcs; 
                lastChunkOffset = index;
                break;
            }
        }    
        /* For the last chunk...*/
    #ifdef GE_DEBUG
        printf("Rank %d: Sending packedTransposeAB last chunk to rank %d\n", myRank, lastChunkProc);
    #endif
        if (lastChunkProc == 0)
        {
            memcpy((&solverState->h_myPartOfPackedTransposeAB[lastChunkOffset * packedCount]), &solverState->h_packedTransposeAB[c * packedCount], solverState->rowCount * lastChunkSize *sizeof(elemtype));
            solverState->myNumChunks++;
            solverState->myLastChunkSize = lastChunkSize;
        }
        else
        {
            MPI_CHECK(MPI_Send((void *)(solverState->h_packedTransposeAB + (c * packedCount)), solverState->rowCount * lastChunkSize * VEC_LEN, MPI_UNSIGNED, lastChunkProc, 0, MPI_COMM_WORLD));
        }
    }
    else
    {
        MPI_Status status;        
        int c;
        int index = 0; 
    #ifdef GE_DEBUG
        printf("Rank %d: Recieving packedTransposeAB", myRank);
    #endif

        c = myRank;
        while (c < (numChunks - 1))
        {

            MPI_CHECK(MPI_Recv((void *)(solverState->h_myPartOfPackedTransposeAB + (index * packedCount)), packedCount * VEC_LEN, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status));
            index++;
            solverState->myNumChunks++;
            c += numProcs;
        }
        if (c == (numChunks - 1))
        {
            MPI_CHECK(MPI_Recv((void *)(solverState->h_myPartOfPackedTransposeAB + (index * packedCount)), solverState->rowCount * lastChunkSize * VEC_LEN, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, &status));
            solverState->myNumChunks++;
            solverState->myLastChunkSize = lastChunkSize;
        }
            
    }
    solverState->myPackedColumnCount = ((solverState->myNumChunks - 1) * CHUNK_SIZE) + solverState->myLastChunkSize;
}


void gatherResult(stSolverState *solverState, int myRank, int nodeLocalRank, int numProcs)
{
    int numChunks, lastChunkSize, lastChunkProc, lastChunkOffset = 0;
    long int packedSize, packedCount;

    numChunks = intCeilDiv(solverState->packedColumnCount, CHUNK_SIZE);
    packedCount = solverState->rowCount * CHUNK_SIZE;
    packedSize = packedCount * sizeof(elemtype);

    // Rank 0 recieves the solution partitions from the other procs
    if (myRank == 0)
    {
        int c = 0, index = 0;
        MPI_Status status;

        lastChunkProc = 0;       
        while (c < (numChunks - 1))
        {
            int r; 

            r = c % numProcs;
        #ifdef GE_DEBUG
            printf("Rank %d: Recieving packedTransposeAB from rank %d\n", myRank, r);
        #endif
            if (r == 0)
            {
                memcpy((&solverState->h_packedTransposeAB[c * packedCount]), (&solverState->h_myPartOfPackedTransposeAB[index * packedCount]), packedSize);
                index++;
            }
            else
            {  
                MPI_CHECK(MPI_Recv((void *)(solverState->h_packedTransposeAB + (c * packedCount)), packedCount * VEC_LEN, MPI_UNSIGNED, r, 0, MPI_COMM_WORLD, &status));
            }
            c++;
            if (c == (numChunks - 1)) 
            {
                lastChunkProc = (r + 1) % numProcs; 
                lastChunkOffset = index;
                break;
            }
        }
        /* For the last chunk...*/
    #ifdef GE_DEBUG
        printf("Rank %d: Recieving packedTransposeAB last chunk from rank %d\n", myRank, lastChunkProc);
    #endif
        if (lastChunkProc == 0)
        {
            memcpy((&solverState->h_packedTransposeAB[c * packedCount]), (&solverState->h_myPartOfPackedTransposeAB[lastChunkOffset * packedCount]), solverState->rowCount * solverState->myLastChunkSize *sizeof(elemtype));
        }
        else
        {
            lastChunkSize = solverState->packedColumnCount - ((numChunks - 1) * CHUNK_SIZE); 
            MPI_CHECK(MPI_Recv((void *)(solverState->h_packedTransposeAB + (c * packedCount)), solverState->rowCount * lastChunkSize * VEC_LEN, MPI_UNSIGNED, lastChunkProc, 0, MPI_COMM_WORLD, &status));
        }
    }
    else
    {
        int c;
        int index = 0; 
    #ifdef GE_DEBUG
        printf("Rank %d: Sending packedTransposeAB", myRank);
    #endif

        c = myRank;
        while (c < (numChunks - 1))
        {

            MPI_CHECK(MPI_Send((void *)(solverState->h_myPartOfPackedTransposeAB + (index * packedCount)), packedCount * VEC_LEN, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD));
            index++;
            c += numProcs;
        }
        if (c == (numChunks - 1))
        {
            MPI_CHECK(MPI_Send((void *)(solverState->h_myPartOfPackedTransposeAB + (index * packedCount)), solverState->rowCount * solverState->myLastChunkSize * VEC_LEN, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD));
        }
            
    }

}
// Please note that this function is called only by Rank 0

void prepareResultFromGaussianElimination(unsigned char* A, unsigned char* B, int rowCount, int columnCount, stSolverState *solverState)
{

    int packedColumnCount;

    packedColumnCount = intCeilDiv(columnCount + 1, PACK_SIZE);

    transposeMatrixCPU(solverState->h_packedAB, solverState->h_packedTransposeAB, packedColumnCount, rowCount);
    unpackLinearSystem(A, B, solverState->h_packedAB, rowCount, columnCount);

    free(solverState->h_packedTransposeAB);
    free(solverState->h_packedAB);
}

void bcastPivotRow(int pivotProc, int myRank, int numProcs, stSolverState *solverState, void **commReq, void **commStatus)
{
    int i;

#ifdef PIVOTPACK
    int packedRowCount = intCeilDiv(solverState->rowCount, PIVOT_PACK_SIZE);
#endif

    //MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (myRank == pivotProc)
    {
        MPI_Request *req;
        MPI_Status *status;
        int tag = 0;

        req = (MPI_Request *)malloc((numProcs - 1)*sizeof(MPI_Request)*2);
        status = (MPI_Status *)malloc((numProcs - 1)*sizeof(MPI_Status)*2);
        for (i=0; i<numProcs; i++)
        {
            if (i == pivotProc)
                continue;
            MPI_CHECK(MPI_Isend((void *)solverState->pivotRowIndex, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &req[tag]));
#ifdef PIVOTPACK
            MPI_CHECK(MPI_Isend((void *)solverState->multipliers, packedRowCount, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD, &req[tag+1]));
#else
            MPI_CHECK(MPI_Isend((void *)solverState->multipliers, solverState->rowCount, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, &req[tag+1]));
#endif
            tag += 2;
        }
        *commReq = req;
        *commStatus = status;
    }
    else
    {
        MPI_Status status[2];
        MPI_Request req[2];
     
        MPI_CHECK(MPI_Irecv((void *)(solverState->pivotRowIndex), 1, MPI_INT, pivotProc, 0, MPI_COMM_WORLD, &req[0]));
#ifdef PIVOTPACK
         MPI_CHECK(MPI_Irecv((void *)(solverState->multipliers), packedRowCount,  MPI_UNSIGNED, pivotProc, 0, MPI_COMM_WORLD, &req[1]));
#else
        MPI_CHECK(MPI_Irecv((void *)(solverState->multipliers), solverState->rowCount, MPI_UNSIGNED_CHAR, pivotProc, 0, MPI_COMM_WORLD, &req[1]));
#endif
        MPI_CHECK(MPI_Waitall(2, req, status));
    }


}

void waitForPivotBcast(int pivotProc, void *req, void *status, int numProcs, int myRank)
{
    if (myRank != pivotProc) return;
    if (numProcs  > 1) MPI_CHECK(MPI_Waitall(2*(numProcs-1), (MPI_Request *)req, (MPI_Status *)status));
    free(req);
    free(status);
}

        
            
