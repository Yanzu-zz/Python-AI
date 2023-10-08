import numpy as np
import sys, os

sys.path.append(os.pardir)
from common.functions import *


# 乘法层
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # 前向传播就正常乘
    # 但是乘法的反向传播需要翻转 x,y 参数
    # 所以我们这里需要用两个变量保存
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    # 反向传播就按书上分析的来
    # 也就是 z=x*y, dz/dx = y, dz/dy=x
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


# 加法层
class AddLayer:
    def __init__(self):
        pass

    # 加法层的前向传播就是简单加上去就行
    # 注意，因为加法层的反向传播也不需要反转什么参数
    # 故我们这里不需要用 x,y 变量记录
    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


# ReLU 函数
# Rectified Linear Unit
class Relu:
    def __init__(self):
        # 用 mask 保存大于小于0的元素位置索引
        self.mask = None

    # 注意，一般传入的 x 是 numpy 数组，不是一个值
    # 不然效率太低
    def forward(self, x):
        # 小于等于 0 的地方的索引
        self.mask = (x <= 0)
        out = x.copy()
        # 一次性把 <=0 的地方全标记上为 0
        # 其余的值就不变
        out[self.mask] = 0

        return out

    def backward(self, dout):
        # 这里也是如果前向传播时值 小于等 0，则终止传播，向后传播 0 值
        dout[self.mask] = 0
        # 否则就简单传递上个节点送来的值
        dx = dout

        return dx


# Sigmoid 函数
class Sigmoid:
    def __init__(self):
        self.out = None

    # 前向传播就是简单的套公式
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    # 而反向传播就需要看书上最终推导出来的式子了
    def backward(self, dout):
        # 也就是 dout*y*(1-y)
        dx = dout * self.out * (1.0 - self.out)
        return dx


# 神经网络的正向传播中进行的矩阵的乘积运算在几何学领域被称为“仿射变换（Affine）”
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    # 前向传播就是简单的矩阵乘法
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        # dout也就是 L对Y求偏导
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


# Softmax 带 交叉熵的实现
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        # softmax的输出
        self.y = None
        # 监督数据（one-hot vector）
        self.t = None

    # 计算 Softmax
    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    # 反向传播的话就是书上的推导
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx