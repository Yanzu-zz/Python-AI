import sys, os

sys.path.append(os.pardir)
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


# 用标准的层来写简单神经网络
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        I, H, O = input_size, hidden_size, output_size
        W1 = np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = np.random.randn(H, O)
        b2 = np.zeros(O)

        # 生成层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        self.loss_layer = SoftmaxWithLoss()

        # 将所有的权重和梯度整理到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    # 也就是正向所有层走一遍就行
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        # 最后单独处理 softmaxwithloss 层
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        # 反向传播就是要 reverse 层从后往前
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
