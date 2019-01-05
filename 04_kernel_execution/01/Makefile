CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=async_execution

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 # --resource-usage

# Gencode argumentes
SMS = 30 35 37 50 52 60 61 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += 
ALL_CCFLAGS += 

all : ${TARGET}

async_execution: async_execution.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

nvprof: async_execution
	nvprof -f -o $+.nvvp ./$+

clean:
	rm -f ${TARGET} *.o
