import numpy as np
from numba import cuda
from timeit import default_timer as timer

@cuda.jit
def matmul(d_c, d_a, d_b):
    x, y = cuda.grid(2)
    if (x < d_c.shape[0] and y < d_c.shape[1]):
        sum = 0
        for k in range(d_a.shape[1]):
            sum += d_a[x, k] * d_b[k, y]
        d_c[x, y] = sum

# initialize input data
N = 8192
a = np.random.rand(N, N).astype(np.float32)
b = np.random.rand(N, N).astype(np.float32)

# copy matrices to the devices
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)

# create device memory for matrix c
d_c = cuda.device_array((N, N))

# configure the blocks
BLOCK_DIM = 16
dimBlock = (BLOCK_DIM, BLOCK_DIM)
dimGrid = (int((N + BLOCK_DIM - 1) / BLOCK_DIM), 
           int((N +BLOCK_DIM - 1) / BLOCK_DIM))

# matrix multiplication (gpu)
start = timer()
matmul[dimGrid, dimBlock](d_c, d_a, d_b)
elapsed_time_gpu = (timer() - start) * 1e3

# copy the result back to the host
c = d_c.copy_to_host()

# matrix multiplication (cpu)
start = timer()
c_host = np.matmul(a, b)
elapsed_time_cpu = (timer() - start) * 1e3

# print elapse times
print("Elapsed Time")
print("GPU: %.3f ms" % elapsed_time_gpu)
print("CPU: %.3f ms" % elapsed_time_cpu)

if (np.allclose(c_host, c)):
    print("Done.")
else:
    print("GPU and host results are mismatching.")

