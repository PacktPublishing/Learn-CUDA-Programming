CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=matrix_transpose conflict_solved

INCLUDES= -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -lineinfo

# Gencode argumentes
SMS = 30 35 37 50 52 60 61 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

all : ${TARGET}

matrix_transpose: matrix_transpose.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@ $<

conflict_solved: conflict_solved.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@ $<

clean:
	rm -f ${TARGET} 
