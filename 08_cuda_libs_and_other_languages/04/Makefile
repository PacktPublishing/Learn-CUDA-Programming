CUDA_PATH=/usr/local/cuda/
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=cufft.1d cufft.half cufft.mgpu
INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 # --resource-usage

# Gencode argumentes
SMS = 60 61 70 # 30 35 37 50 52
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += -lcufft
ALL_CCFLAGS += -std=c++11 $(INCLUDES) -L/usr/local/cuda/lib

all : ${TARGET}

cufft.1d: cufft.1d.cpp
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

cufft.half: cufft.half.cpp
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

complex.o: complex.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

cufft.mgpu: cufft.mgpu.cpp complex.o
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

nvprof: cufft.1d
	nvprof -f -o $+.nvvp ./$+

clean:
	rm -f ${TARGET} *.o
