import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int32)


# 跃迁函数
x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
plt.plot(x, y1)
plt.ylim(-0.1, 1.1)  # 指定 y 轴范围


# plt.show()


# sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


y2 = sigmoid(x)
plt.plot(x, y2)
plt.ylim(-0.1, 1.1)


# plt.show()


# ReLU 函数
def ReLU(x):
    return np.maximum(0, x)


# 恒等函数（为了保持和其它激活函数相同格式）
def indentity_function(x):
    return x


# 简单3层神经网络模拟练习
def init_network():
    # 定义权重和偏置值
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])

    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


# 前向传播函数，也就是从输入到输出方向的传递处理
def forward(network, x):
    # 也就是 值*权重+偏置 传给下个神经元节点
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 得到第1层神经节点值（第0层为输入点）
    a1 = np.dot(x, W1) + b1
    # 接着执行激活函数
    z1 = sigmoid(a1)

    # 下面就是重复 forward 操作了，直到输出节点
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = indentity_function(a3)

    return y


network = init_network()
# 输出值
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)


# 这样实现容易溢出
def softmax_bad(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# 通过数学推导可知，分子分母求指数时候同时加减一个数C不会影响值
# 故我们选用 a 钟最大的值减去他即可
def softmax(a):
    C = np.max(a)
    # 放溢出
    exp_a = np.exp(a - C)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
