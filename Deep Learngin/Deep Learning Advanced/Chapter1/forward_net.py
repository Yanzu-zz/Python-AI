import numpy as np


class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


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
