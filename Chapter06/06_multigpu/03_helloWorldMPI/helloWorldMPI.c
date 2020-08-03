#include <mpi.h>
#include <stdio.h>
int main(int argc, char *argv[]) {
        int rank,size;
        /* Initialize the MPI library */
        MPI_Init(&argc,&argv);
        /* Determine the calling process rank and total number of ranks */
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        MPI_Comm_size(MPI_COMM_WORLD,&size);
        /* Compute based on process rank */
        /* Call MPI routines like MPI_Send, MPI_Recv, ... */
        printf("\n Rank %d, Size %d", rank,size);
        /* Shutdown MPI library */
                MPI_Finalize();
        return 0;
}

