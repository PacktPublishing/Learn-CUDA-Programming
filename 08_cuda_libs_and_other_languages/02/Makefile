CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=cublasGemmEx.fp32 cublasGemmEx.fp16 cublasGemmEx.fp16.perf cublasGemmEx.int8

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 # --resource-usage

# Gencode argumentes
# SM for 30 35 37 50 is not available INT8 and FP16 52 60 61 
SMS = 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += -lcublas
ALL_CCFLAGS += -std=c++11 $(INCLUDES) -L/usr/local/cuda/lib

all : ${TARGET}

cublasGemmEx.fp32: cublasGemmEx.fp32.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

cublasGemmEx.fp16: cublasGemmEx.fp16.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

cublasGemmEx.fp16.perf: cublasGemmEx.fp16.perf.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

cublasGemmEx.int8: cublasGemmEx.int8.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

nvprof: cublas
	nvprof -f -o $+.nvvp ./$+

clean:
	rm -f ${TARGET} *.o
