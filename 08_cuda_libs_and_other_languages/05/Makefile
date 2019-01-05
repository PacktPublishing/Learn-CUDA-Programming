CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=blur blur_stream

INCLUDES = -I${CUDA_PATH}/samples/common/inc `pkg-config --cflags opencv`
NVCC_FLAGS=-m64 # --resource-usage

# Gencode argumentes
SMS = 60 61 70 # 30 35 37 50 52
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += `pkg-config --libs opencv`
ALL_CCFLAGS += -std=c++11 $(INCLUDES) -L/usr/local/cuda/lib

all : ${TARGET}

blur: blur.cpp
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

blur_stream: blur_stream.cpp
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

nvprof: blur
	nvprof -f -o $+.nvvp ./$+

clean:
	rm -f ${TARGET} *.o
