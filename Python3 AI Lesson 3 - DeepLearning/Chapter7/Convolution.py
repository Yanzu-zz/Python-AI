from collections import OrderedDict

import numpy as np
import sys, os

sys.path.append(os.pardir)
from common.util import im2col
from common.layers import *


# 卷积神经网络
class ConvolutionDemo:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        # 计算输出的矩阵格式
        # 也就是经过滤波器之后的长宽
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # 变成矩阵形状（矩阵加速的优化已经很成熟）
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W)

        # 重新变回高维
        # transpose 可以改变维的顺序
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out


class PoolingDemo:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        # 计算经过池化滤波器后后的长宽
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 打平
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # 然后弄成二维形式
        col = col.reshape(-1, self.pool_h * self.pool_w)

        out = np.max(col, axis=1)
        # 换回高维形式给下一层
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out


# 搭建一个简单的 CNN 网络
# CNN 的构成是：Convolution - ReLU - Pooling - Affine - ReLU - Affine - Softmax -> output
# ------- 参数 --------
# input_dim―输入数据的维度：（通道，高，长）
# conv_param―卷积层的超参数（字典）。字典的关键字如下：
#   filter_num―滤波器的数量
#   filter_size―滤波器的大小
#   stride―步幅
#   pad―填充
# hidden_size―隐藏层（全连接）的神经元数量
# output_size―输出层（全连接）的神经元数量
# weitght_int_std―初始化时权重的标准差
class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # 权重参数的初始化
        # 也就是 forward 时需要用到的参数
        self.params = {}
        # W1，b1 是卷积层的参数
        self.params['W1'] = weight_init_std * np.random.rand(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        # 第二个全连接层
        self.params['W2'] = weight_init_std * np.random.rand(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        # 第三个全连接层
        self.params['W3'] = weight_init_std * np.random.rand(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 然后生成必要的层
        self.layers = OrderedDict()
        # 卷积层
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        # ReLU 激活函数层
        self.layers['Relu1'] = Relu()
        # 池化
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        # Affine 矩阵运算
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])

        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    # 预测方法，也就是正向走一遍神经网络
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    # 反向传播法计算梯度
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 把各个权重参数的梯度保存到grads字典中
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db
        return grads
