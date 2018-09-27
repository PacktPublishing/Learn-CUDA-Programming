CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
MPICC ?= mpicc
TARGET=cuda_kernel

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -rdc=true # --resource-usage

# Gencode argumentes
SMS = 35 37 50 52 60 61 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += 
ALL_CCFLAGS += ${NVCC_FLAGS}

all : ${TARGET}

cuda_kernel: cuda_kernel.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) $(INCLUDES)

nvprof: cuda_kernel
	nvprof -f -o $+.nvvp ./$+

clean:
	rm -f ${TARGET} *.o *.nvvp
