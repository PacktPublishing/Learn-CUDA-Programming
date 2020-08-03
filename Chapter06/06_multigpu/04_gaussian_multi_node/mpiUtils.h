#ifndef MPIUTILS_H
#define MPIUTILS_H

#include <mpi.h>

#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { \
        printf("MPI error calling \"%s\"\n", #call); \
        MPI_Abort(MPI_COMM_WORLD, -1); }

#endif


