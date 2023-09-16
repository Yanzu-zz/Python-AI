import numpy as np
from negative_sampling_layer import *

import sys, os

sys.path.append(os.pardir)


# 改进后的 CBOW
# 也就是输出层 MatMul 层替换成 Embedding
# 中间和后面的 softmax 多分类换成 sigmoid 二分类与负采样
class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 初始化权重
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 生成层
        self.in_layers = []
        # 这样写就可以兼容 window_size > 1 的情况
        # 也就是可以有多个上下文
        for i in range(window_size << 1):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        # 此时这里是二分类 sigmoid+负采样
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 将所有权重和梯度整理到列表中
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.parms += layer.params
            self.grads += layer.grads

        # 将单词的分布式表示设置为成员变量
        self.word_vecs = W_in

    # 正向传播就是正常的走一遍神经网络
    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        # 取所有层的平均值
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)

        return loss

    # 反向传播就是顺序要和正向传播相反即可
    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)

        return None
