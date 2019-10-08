#!/bin/bash
MPI_VERSION="3.0.4"

wget -O /tmp/openmpi-${MPI_VERSION}.tar.gz https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-${MPI_VERSION}.tar.gz
tar xzf /tmp/openmpi-${MPI_VERSION}.tar.gz -C /tmp
cd /tmp/openmpi-${MPI_VERSION}
./configure --enable-orterun-prefix-by-default
make -j $(nproc) all && sudo make install
sudo ldconfig
mpirun --version