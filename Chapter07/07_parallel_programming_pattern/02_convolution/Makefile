CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=convolution

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS= -lineinfo --maxrregcount=48 --resource-usage -Xcompiler -rdynamic -Xcompiler -fopenmp -rdc=true

IS_CUDA_11:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0}

# Gencode argumentes
SMS = 35 37 50 52 60 61 70 75
ifeq "$(IS_CUDA_11)" "1"
SMS = 52 60 61 70 75 80
endif
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += -L/usr/local/cuda/lib -lgomp
ALL_CCFLAGS += -m64 -g -std=c++11 $(NVCC_FLAGS) $(INCLUDES) $(LIBRARIES)

all : ${TARGET}

convolution: convolution.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

nvprof: convolution
	nvprof -f -o $+.nvvp --profile-from-start off ./$+
	nvprof -f -o $+-metrics.nvvp --analysis-metrics ./$+

clean:
	rm -f ${TARGET} *.o *.nvvp
