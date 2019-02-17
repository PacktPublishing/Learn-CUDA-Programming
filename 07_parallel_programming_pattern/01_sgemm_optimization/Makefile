CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=sgemm_original sgemm_optimized

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 # --resource-usage

# Gencode argumentes
SMS = 30 35 37 50 52 60 61 70 75
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += -lgomp
ALL_CCFLAGS += -std=c++11 -Xcompiler -fopenmp -rdc=true $(INCLUDES)

all : ${TARGET}
	
sgemm_original: sgemm_original.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

sgemm_optimized: sgemm_optimized.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

nvprof: sgemm_original
	nvprof -f -o $+.nvvp ./$+

clean:
	rm -f ${TARGET} *.o
