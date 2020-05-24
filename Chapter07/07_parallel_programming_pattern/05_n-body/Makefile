CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=n-body

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 # --resource-usage

# Gencode argumentes
SMS = 30 35 37 50 52 60 61 70 75
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIBRARIES += -L/usr/local/cuda/lib 
ALL_CCFLAGS += -std=c++11 $(INCLUDES) $(LIBRARIES)

all : ${TARGET}

n-body: n_body.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+

nvprof: n-body
	nvprof -f -o $+.nvvp --profile-from-start off ./$+
	nvprof -f -o $+-metrics.nvvp --analysis-metrics ./$+

clean:
	rm -f ${TARGET} *.o *.nvvp
