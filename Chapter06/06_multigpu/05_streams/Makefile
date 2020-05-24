CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=vector_addition merging_multi_gpu

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -lineinfo --default-stream per-thread -Xcompiler -fopenmp # --resource-usage 

LIBRARIES= -lgomp

# Gencode argumentes
SMS = 30 35 37 50 52 60 61 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

all: ${TARGET}

vector_addition: vector_addition.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

merging_multi_gpu: image_merging.cu scrImagePgmPpmPackage.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f ${TARGET} *.o
