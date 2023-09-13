import numpy as np


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    # 前向传播也是简单矩阵相乘
    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        # b 是 repeat 扩展过的
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


# Matrix Multiply
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    # 前向传播就是简单的矩阵乘法
    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    # 反向传播需要推导公式
    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.t)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


# 复习简单两层神经网络
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏重
        I, H, O = input_size, hidden_size, output_size
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 初始化（生成）层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 将参数加入变量中
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    # 将 x 在神经网络内走一趟
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x


# 测试一下我们的简单两层模型
x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
print(s)
