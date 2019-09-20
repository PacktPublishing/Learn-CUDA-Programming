CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=image_histogram

INCLUDES = 
NVCC_FLAGS=-m64 -lineinfo #--resource-usage

# Gencode argumentes
SMS = 70 75
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS += -rdc=true $(NVCC_FLAGS) $(INCLUDES)

all : ${TARGET}
	
image_histogram: scrImagePgmPpmPackage.cpp image_histogram.cu
        $(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $+ $(LIBRARIES)
        $(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -run -o $@ $+

clean:
	rm -f ${TARGET} *.o
