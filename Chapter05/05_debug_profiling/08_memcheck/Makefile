CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=simple_sgemm simple_sgemm_oob simple_sgemm_mem_leak

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64

# Gencode argumentes
SMS = 30 35 37 50 52 60 61 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += -lgomp

ALL_CCFLAGS += -g -G -Xcompiler -rdynamic $(NVCC_FLAGS) $(INCLUDES)

all : ${TARGET}

simple_sgemm: simple_sgemm.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

simple_sgemm_oob: simple_sgemm_oob.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

simple_sgemm_mem_leak: simple_sgemm_mem_leak.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f ${TARGET} *.o
