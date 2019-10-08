CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=dynamic_parallelism recursion

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -rdc=true -lcudadevrt # --resource-usage

# Gencode argumentes
SMS = 35 37 50 52 60 61 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += 
ALL_CCFLAGS += ${NVCC_FLAGS}

all : ${TARGET}

dynamic_parallelism: dynamic_parallelism.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) $(INCLUDES)

recursion: recursion.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) $(INCLUDES)

nvprof: dynamic_parallelism
	nvprof -f -o $+.nvvp ./$+

clean:
	rm -f ${TARGET} *.o *.nvvp
