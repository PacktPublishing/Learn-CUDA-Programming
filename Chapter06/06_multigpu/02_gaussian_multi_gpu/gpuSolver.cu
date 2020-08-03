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
	gaussianEliminationMultiGPU(packedTransposeAB, rowCount, columnCount);
	transposeMatrixCPU(packedAB, packedTransposeAB, packedColumnCount, rowCount);
	unpackLinearSystem(A, B, packedAB, rowCount, columnCount);
	return;
}

void gaussianEliminationMultiGPU(unsigned int* packedTransposeAB, int rowCount, int columnCount)
{
	int i, part, partId;
	int currentPartPackedColumnCount, lastPartPackedColumnCount;
	long int offsetAB;
	int packedColumnsPerPart;
	int packedColumnCount;

	stSolverState solverState[NUMBER_OF_GPU];

	packedColumnCount         = intCeilDiv(columnCount + 1, PACK_SIZE);
	packedColumnsPerPart      = intCeilDiv(packedColumnCount, NUMBER_OF_GPU);
	lastPartPackedColumnCount = packedColumnCount - (packedColumnsPerPart * (NUMBER_OF_GPU - 1));

	long int packedSize   = sizeof(unsigned int)*rowCount*packedColumnsPerPart;


	if(setupPeerToPeer(NUMBER_OF_GPU) != 1 && testPeerToPeer(NUMBER_OF_GPU))
	{
		printf("Failed setting up peer to peer access. Exiting\n");
		exit(-1);
	}


	for(part = 0; part < NUMBER_OF_GPU; part++)
	{
		checkCudaErrors(cudaSetDevice(part));
		offsetAB = ((long int)part) * rowCount * packedColumnsPerPart;
		currentPartPackedColumnCount = (part == (NUMBER_OF_GPU - 1))? lastPartPackedColumnCount : packedColumnsPerPart;
		initializeSolver(&solverState[part], rowCount, currentPartPackedColumnCount*PACK_SIZE-1, &packedTransposeAB[offsetAB]);

	}

	for(i = 0; i < columnCount; i++)
	{
		partId                  = (i >> 5) / packedColumnsPerPart;
		solverState[partId].columnPack  = (i >> 5) % packedColumnsPerPart;
		solverState[partId].columnBit   = i % PACK_SIZE;
		checkCudaErrors(cudaSetDevice(partId));
		launchFindPivotRowKernel(&solverState[partId]);
		checkCudaErrors(cudaDeviceSynchronize());
		bcastPivotRow(partId, 0, NUMBER_OF_GPU, solverState);

		for(part = 0; part < NUMBER_OF_GPU; part++)
		{
			checkCudaErrors(cudaSetDevice(part));
			solverState[part].columnPack  = (i >> 5) - part*packedColumnsPerPart;
			solverState[part].columnBit   = i % PACK_SIZE;
			launchExtractPivotRowKernel(&solverState[part]);
			launchRowEliminationKernel(&solverState[part]);
			checkCudaErrors(cudaDeviceSynchronize());
		}
	}

	for(part = 0; part < NUMBER_OF_GPU; part++)
	{    
		offsetAB = ((long int)part) * rowCount * packedColumnsPerPart;
		gatherResult(&packedTransposeAB[offsetAB], &solverState[part]);
	}

	for(part = 0; part < NUMBER_OF_GPU; part++)
	{
		freeSolver(&solverState[part]);
	}

	for(part = 0; part < NUMBER_OF_GPU; part++)
	{
		checkCudaErrors(cudaSetDevice(part));
		checkCudaErrors(cudaDeviceSynchronize());
	}
}

void bcastPivotRow(int pivotPart, int myPart, int numGPUs, stSolverState *solverState) /* myPart is ignored in multinode p2p*/
{
	int part; 

	for(part = 0; part < numGPUs; part++)
	{
		if(part == pivotPart)
			continue;

		checkCudaErrors(cudaMemcpyPeer(solverState[part].pivotRowIndex, part, solverState[pivotPart].pivotRowIndex,
					pivotPart, sizeof(int)));
		checkCudaErrors(cudaMemcpyPeer(solverState[part].multipliers,   part, solverState[pivotPart].multipliers,
					pivotPart, sizeof(unsigned char)*solverState[pivotPart].rowCount));
	}
}
