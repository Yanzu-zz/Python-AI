import numpy as np
import sys, os

sys.path.append(os.pardir)
from common.layers import MatMul

a = np.array([[0, 1, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 1, 1, 0],
              [0, 1, 0, 1, 0, 0, 0],
              [0, 0, 1, 0, 1, 0, 0],
              [0, 1, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 1, 0]], dtype=np.int32)

s = np.sum(a, axis=0)
# print(s)

c = np.array([[1, 0, 0, 0, 0, 0, 0]])
# print(c.shape)
W = np.random.randn(7, 3)
# print(W.shape)
h = np.dot(c, W)
# print(h)

# 或者可以用之前写过的矩阵乘法 MatMul 层实现
Mlayer = MatMul(W)
h = Mlayer.forward(c)
# print(h)


arr1 = np.arange(21).reshape(7,3)
print(arr1)

out1 = np.array([0,5])
print(arr1[out1])