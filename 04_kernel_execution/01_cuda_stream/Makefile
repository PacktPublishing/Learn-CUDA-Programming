CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=cuda_default_stream cuda_multi_stream cuda_multi_stream_with_sync cuda_multi_stream_with_default

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 # --resource-usage

# Gencode argumentes
SMS = 30 35 37 50 52 60 61 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += 
ALL_CCFLAGS += 

all : ${TARGET}

cuda_default_stream: 1_cuda_default_stream.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIBRARIES)

cuda_multi_stream: 2_cuda_multi_stream.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIBRARIES)

cuda_multi_stream_with_sync: 3_cuda_multi_stream_with_sync.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIBRARIES)

cuda_multi_stream_with_default: 4_cuda_multi_stream_with_default.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIBRARIES)
	
clean:
	rm -f ${TARGET} *.o *.nvvp
