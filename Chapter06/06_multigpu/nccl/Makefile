CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=nccl

INCLUDES = -I${CUDA_PATH}/samples/common/inc -I./utils
NVCC_FLAGS=--resource-usage -Xcompiler -rdynamic -Xcompiler -fopenmp -rdc=true 

# Gencode argumentes
SMS = 30 35 37 50 52 60 61 70 75
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += -L/usr/local/cuda/lib -lnccl -lgomp
ALL_CCFLAGS += -m64 -g -std=c++11 $(NVCC_FLAGS) $(INCLUDES) $(LIBRARIES)

OBJ_DIR = obj

all : ${TARGET}

nccl: nccl.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

nvprof: train
	nvprof -f -o $+.nvvp ./$+

clean:
	rm -rf ${OBJ_DIR}
	rm -f ${TARGET} *.o

