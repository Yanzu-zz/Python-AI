import numpy as np


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    # 当梯度大于阈值的时候
    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


# 循环神经网络（Recurrent Neural Network）
class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    # RNN 的前向传播就是根据上个节点传来的数据（h_prev）与相应权重
    # 以及本个节点传入的x和权重w进行计算（当然可能还有偏置）
    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        # tanh 的反向传播公式
        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)

        # 下面是两个矩阵乘法的反向传播（之前推过）
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.t, dt)
        dx = np.dot(dt, Wx.T)

        # 将获得的参数写进 grads，以便继续优化
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


# 一次处理 T 个 RNN 层
class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self, dh = None, None
        self.stateful = stateful

    # 状态管理
    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)
            # 按顺序对第 t 个层进行正向传播
            # 第一个参数 xs 也就是按顺序排列的输入数据
            # 第二个参数 self.h 就是上个层的输出
            self.h = layer.forward(xs[:, t, :], self.h)
            # 这里的 self.h 是本层的输出结果
            hs[:, t, :] = self.h
            self.layers.append(layer)

        # 返回 T 个 RNN 层的输出
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]

        for t in reversed(range(T)):
            layer = self.layers[t]
            # 按照顺序进行反向传播
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

            for i, grad in enumerate(grads):
                self.grads[i][...] = grad
            self.dh = dh

            return dh
