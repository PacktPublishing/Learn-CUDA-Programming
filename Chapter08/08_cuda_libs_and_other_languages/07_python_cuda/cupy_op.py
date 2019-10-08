import numpy as np
import cupy as cp

# cupy matmul
a = cp.random.uniform(0, 1, (2, 4)).astype('float32')
b = cp.random.uniform(0, 1, (4, 2)).astype('float32')
c = cp.matmul(a, b)
print("Matrix Multiplication")
print("a::\n", a)
print("b::\n", b)
print("c = a' * b::", c)

# custom kernel
squared_diff = cp.ElementwiseKernel(
    'float32 x, float32 y',
    'float32 z',
    'z = (x - y) * (x - y)',
    'squared_diff')

a = cp.random.uniform(0, 1, (2, 4)).astype('float32')
b = cp.random.uniform(0, 1, (2, 4)).astype('float32')
c = squared_diff(a, b)
print("Elements Diff")
print("a::\n", a)
print("b::\n", b)
print("c = (a-b)*(a-b)::", c)
