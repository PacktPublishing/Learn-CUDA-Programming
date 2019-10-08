#!/usr/bin/python3

# initialize the device
import pycuda.autoinit

from pycuda import driver, compiler, gpuarray
import numpy as np
from string import Template

import timeit

kernel_code_template = Template("""
__global__ void matmul_kernel(float *d_C, float *d_A, float *d_B)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.f;
    for (int e = 0; e < ${MATRIX_SIZE}; e++)
        sum += d_A[idx_y * ${MATRIX_SIZE} + e] * d_B[e * ${MATRIX_SIZE} + idx_x];
    d_C[idx_y * ${MATRIX_SIZE} + idx_x] = sum;
}
""")

N = 8192
np.random.seed(2019)
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

# update kernel size with N
# kernel_code = kernel_code_template % { 'MATRIX_SIZE' : N }

# compile the kernel code
mod = compiler.SourceModule( \
        kernel_code_template.substitute(MATRIX_SIZE=N))

# get the kernel function from the compiled module
matmul_kernel = mod.get_function("matmul_kernel")

dimBlock = 16
dimGrid = int((N + dimBlock - 1) / dimBlock)

# prepare to get the gpu events
start = driver.Event()
stop = driver.Event()

# call function
print("Started GPU operation...")
start.record()

matmul_kernel(driver.Out(C), driver.In(A), driver.In(B), 
            block=(dimBlock, dimBlock, 1), 
            grid=(dimGrid, dimGrid))

stop.record()
stop.synchronize()
gpu_time = stop.time_since(start)
print("GPU Execution Time: %.3f ms" % (gpu_time))

print("Started Host operation...")
start = timeit.default_timer()
c_host = np.matmul(A, B)
host_time = timeit.default_timer() - start

print("CPU Execution Time: %.3f ms" % (host_time * 1e3))

if (np.allclose(c_host, C)):
    print("Done.")
else:
    print("GPU and host results are mismatching.")
