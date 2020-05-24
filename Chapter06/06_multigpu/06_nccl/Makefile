CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=nccl

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -lineinfo -lnccl # --resource-usage  

# Gencode argumentes
SMS = 30 35 37 50 52 60 61 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

all: ${TARGET}

nccl: nccl.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f ${TARGET} *.o
