CC = g++
NVCC = nvcc

ROWS ?= 11000
COLS ?= 10000

PARAMS = -DROWS=$(ROWS) -DCOLS=$(COLS)

CFLAGS = -O3 $(PARAMS) -I$(CUDA_PATH)/include/ 
#NV_CFLAGS = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60 -O3 --ptxas-options=-v -lineinfo $(PARAMS) -I$(CUDA_PATH)/include/ -Wno-deprecated-gpu-targets
NV_CFLAGS = -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -O3 -lineinfo $(PARAMS) -I$(CUDA_PATH)/include/ -Wno-deprecated-gpu-targets

BINARY = gaussian_multi_gpu_p2p.out
all: $(BINARY)

OBJECTS = gaussian_multi_gpu_p2p.o

$(BINARY): $(OBJECTS)
	$(NVCC) $(NV_CFLAGS) -dlink $(OBJECTS) -o  gpuObjectCode.o
	$(NVCC) $(NV_CFLAGS) gpuObjectCode.o $(OBJECTS) -o $(BINARY)

%.o : %.c #default rule for making .o files from .c
	$(info --- Building '$@' from '$<' using default rule 1)
	$(CC)  $(CFLAGS) -c -o $@ $<

%.o : %.cu #default rule for making .o files from .cu
	$(info --- Building '$@' from '$<' using default rule 2)
	$(NVCC) $(NV_CFLAGS) -dc -o $@ $<

clean:
	rm -rf *o $(BINARY)

run: $(BINARY)
	./$(BINARY)
