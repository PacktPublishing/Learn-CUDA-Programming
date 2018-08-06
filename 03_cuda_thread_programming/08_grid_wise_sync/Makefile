CUDA_PATH=/usr/local/cuda
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
TARGET=reduction_grid_sync

INCLUDES = -I${CUDA_PATH}/samples/common/inc
NVCC_FLAGS=-m64 -lineinfo # --resource-usage

# Gencode argumentes
SMS = 60 61 70 # 30 35 37 50 52 60 61 70
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += -lgomp
ALL_CCFLAGS += -g -Xcompiler -fopenmp -rdc=true

all : ${TARGET}

reduction_grid_sync_kernel.o: reduction_grid_sync_kernel.cu
	$(EXEC) $(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction_grid_sync.o: reduction_grid_sync.cpp
	$(EXEC) $(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

reduction_grid_sync: reduction_grid_sync.o reduction_grid_sync_kernel.o
	$(EXEC) $(NVCC) $(NVCC_FLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)


clean:
	rm -f ${TARGET} *.o
