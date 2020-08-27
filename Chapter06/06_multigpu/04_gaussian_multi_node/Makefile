CC = g++
NVCC = nvcc
MPICC = mpiCC

MPIRUN = mpirun
CUDA_PATH = /usr/local/cuda

ROWS ?= 11000
COLS ?= 10000

PARAMS = -DROWS=$(ROWS) -DCOLS=$(COLS)

HOSTFILE ?= myHosts

CFLAGS = -O3 $(PARAMS) -I$(CUDA_PATH)/include/ -I$(MPI_PATH)/include 
INCLUDES = -I$(CUDA_PATH)/include/ -I$(CUDA_PATH)/samples/inc -I$(CUDA_PATH)/targets/x86_64-linux/include

IS_CUDA_11:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0}

# Gencode argumentes
SMS = 35 37 50 52 60 61 70 75
ifeq "$(IS_CUDA_11)" "1"
SMS = 52 60 61 70 75 80
endif
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

NV_CFLAGS = $(GENCODE_FLAGS) -O3 -lineinfo $(PARAMS) $(INCLUDES) -Wno-deprecated-gpu-targets

BINARY = gaussian_multi_gpu_rdma.out
all: $(BINARY)

OBJECTS := gaussian_multi_gpu_rdma.o gpuSolver.o gpuSolverFunctions.o linearSystemOps.o utilities.o elementUtilities.o

$(BINARY): $(OBJECTS)
	$(NVCC) $(NV_CFLAGS) -dlink $(OBJECTS) -o  gpuObjectCode.o
	$(MPICC) gpuObjectCode.o $(OBJECTS) -o $(BINARY) -lcudart -L $(CUDA_PATH)/lib64/

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
