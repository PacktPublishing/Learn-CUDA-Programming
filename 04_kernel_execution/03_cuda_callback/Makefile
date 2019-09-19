CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=cuda_callback

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 --default-stream per-thread # --resource-usage

# Gencode argumentes
SMS = 30 35 37 50 52 60 61 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += 
ALL_CCFLAGS += $(NVCC_FLAGS)

all : ${TARGET}

cuda_callback: cuda_callback.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) $(INCLUDES)

nvprof: cuda_callback
	nvprof -f -o $+_${STREAMS}.nvvp --cpu-thread-tracing on ./$+ ${STREAMS}

clean:
	rm -f ${TARGET} *.o *.nvvp
