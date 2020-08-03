CC = g++
NVCC = nvcc
MPICC = mpiCC

MPIRUN = mpirun

ROWS ?= 11000
COLS ?= 10000

PARAMS = -DROWS=$(ROWS) -DCOLS=$(COLS)

HOSTFILE ?= myHosts

CFLAGS = -O3 $(PARAMS) -I$(CUDA_PATH)/include/ -I$(MPI_PATH)/include 
#NV_CFLAGS = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -O3 -lineinfo $(PARAMS) -I$(CUDA_PATH)/include/ -Wno-deprecated-gpu-targets
NV_CFLAGS = -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -O3 -lineinfo $(PARAMS) -I$(CUDA_PATH)/include/ -Wno-deprecated-gpu-targets

BINARY = gaussian_multi_gpu_rdma.out
all: $(BINARY)

OBJECTS = gaussian_multi_gpu_rdma.o gpuSolver.o gpuSolverFunctions.o linearSystemOps.o utilities.o elementUtilities.o

$(BINARY): $(OBJECTS)
	$(NVCC) $(NV_CFLAGS) -dlink $(OBJECTS) -o  gpuObjectCode.o
	$(MPICC)  gpuObjectCode.o $(OBJECTS) -o $(BINARY) -lcudart -L $(CUDA_PATH)/lib64/

%.o : %.c #default rule for making .o files from .c
	$(info --- Building '$@' from '$<' using default rule 1)
	$(MPICC)  $(CFLAGS) -c -o $@ $<

%.o : %.cu #default rule for making .o files from .cu
	$(info --- Building '$@' from '$<' using default rule 2)
	$(NVCC) $(NV_CFLAGS) -dc -o $@ $<

clean:
	rm -rf *o $(BINARY)

run: $(BINARY)
	$(MPIRUN) --hostfile $(HOSTFILE) ./$(BINARY) $(ROWS) $(COLS)
