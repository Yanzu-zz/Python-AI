import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    # W = W + v
    # v = αv - η(dL/dW) （这里 v 可以看作是物理上的速度量）
    # 这里 momentum 就是公式中的 α
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        # 注意，v 是个矩阵
        self.v = None

    # 跟新参数
    def update(self, params, grads):
        # 初始化 v
        # v会以字典型变量的形式保存与参数结构相同的数据
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 若已经初始化过了，则直接根据公式更新参数即可
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# Adaptive Gradient
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        # 空的话就初始化成相同大小的字典
        if self.h i None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        # 否则就进行参数更新操作
        for key in params.keys():
            # 根据书上的公式来即可
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        # 训练阶段才会有删除节点操作
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask
