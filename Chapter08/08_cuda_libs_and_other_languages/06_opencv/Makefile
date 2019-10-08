CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=test blur blur_stream

INCLUDES = -I${CUDA_PATH}/samples/common/inc $(shell pkg-config opencv4 --cflags)
NVCC_FLAGS=-m64 # --resource-usage

# Gencode argumentes
SMS = 60 61 70 75 # 30 35 37 50 52
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += -L/usr/local/cuda/lib -L/usr/local/lib $(shell pkg-config opencv4 --libs)
ALL_CCFLAGS += -std=c++11 $(INCLUDES) $(LIBRARIES)

all : ${TARGET}

test: test.cpp
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

blur: blur.cpp
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

blur_stream: blur_stream.cpp
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

nvprof: blur
	nvprof -f -o $+.nvvp ./$+

clean:
	rm -f ${TARGET} *.o
