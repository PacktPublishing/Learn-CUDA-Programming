import numpy as np
from numba import vectorize
from timeit import default_timer as timer

@vectorize(["float32(float32, float32, float32)"], target='cuda')
def saxpy_cuda(scala, a, b):
    return scala * a + b


#@vectorize(["float32(float32, float32, float32)"], target='cpu')
@vectorize(["float32(float32, float32, float32)"], target='parallel')
def saxpy_host(scala, a, b):
    return scala * a + b

scala = 2.0
np.random.seed(2019)

print("size \t\t CUDA \t\t CPU")
for i in range(16,20):
    N = 1 << i
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)
    c = np.zeros(N, dtype=np.float32)

    # warm-up
    c = saxpy_cuda(scala, a, b)

    # measuring execution time
    start = timer()
    c = saxpy_host(scala, a, b)
    elapsed_time_host= (timer() - start) * 1e3

    start = timer()
    c = saxpy_cuda(scala, a, b)
    elapsed_time_cuda = (timer() - start) * 1e3

    print("[%d]: \t%.3f ms\t %.3f ms" % (N, elapsed_time_cuda, elapsed_time_host))
