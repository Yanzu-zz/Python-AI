import sys, os
import pickle
import numpy as np

sys.path.append(os.pardir)
from data_processing.mnist import load_mnist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a - C)
    sum_exp_a = np.sum(exp_a)

    return exp_a / sum_exp_a


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


# 开启神经网络计算
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


# 单任务
def single_task():
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        # 获取概率最高的元素的索引
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1

    print("Accuracy: " + str(float(accuracy_cnt) / len(x)))


# 批处理
def batch_task():
    x, t = get_data()
    network = init_network()

    # 一次处理100张图片
    batch_size = 100
    accuracy_cnt = 0

    # 步长为 batch_size=100
    for i in range(0, len(x), batch_size):
        # 一次取 batch_size=100 张图片
        x_batch = x[i:i + batch_size]
        y_batch = predict(network, x_batch)
        # axis=1 沿着第一维（每行）寻找最大预测结果
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i + batch_size])

    print("Accuracy: " + str(float(accuracy_cnt) / len(x)))

print("singe task: ")
single_task()

print("multiple(batch) task: ")
batch_task()
