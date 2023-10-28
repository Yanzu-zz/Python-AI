import sys, os

sys.path.append(os.pardir)
import numpy as np
from data_processing.mnist import load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

batch_size = 10
train_size = x_train.shape[0]
# 随机抽取 batch_size 个数据
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
y_batch = y_train[batch_mask]


# print(batch_mask)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 这样单个数据和batch_size 个数据一起传进来都能很好计算交叉熵
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
