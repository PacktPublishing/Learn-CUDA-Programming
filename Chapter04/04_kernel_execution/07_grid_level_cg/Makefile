CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=reduction

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -lineinfo # --resource-usage

IS_CUDA_11:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0}

# Gencode argumentes
SMS = 35 37 50 52 60 61 70 75
ifeq "$(IS_CUDA_11)" "1"
SMS = 52 60 61 70 75 80
endif
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += -lgomp
ALL_CCFLAGS += -g -Xcompiler -fopenmp -rdc=true

all : ${TARGET}

reduction_kernel.o: reduction_kernel.cu
	$(EXEC) $(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction.o: reduction.cpp
	$(EXEC) $(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction: reduction.o reduction_kernel.o
	$(EXEC) $(NVCC) $(NVCC_FLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

nvprof: reduction
	nvprof -f -o $+.nvvp --cpu-thread-tracing on ./$+
	nvprof -f -o $+-metric.nvvp --analysis-metrics ./$+

clean:
	rm -f ${TARGET} *.o
