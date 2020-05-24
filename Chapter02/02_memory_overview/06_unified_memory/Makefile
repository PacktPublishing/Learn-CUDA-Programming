CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=unified_simple unified_initialized unified_prefetch unified_64align

INCLUDES= -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -lineinfo

# Gencode argumentes
SMS = 30 35 37 50 52 60 61 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

all : ${TARGET}

unified_simple: unified_memory.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@.out $<

unified_initialized: unified_memory_initialized.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@.out $<

unified_prefetch: unified_memory_prefetch.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@.out $<

unified_64align: unified_memory_64align.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@.out $<

clean:
	rm -f *.out 
