from two_layer_net import *
import matplotlib.pyplot as plt

import sys, os

sys.path.append(os.pardir)
import numpy as np
# from common.optimizer import SGD
from dataset import spiral
from common.trainer import Trainer

""" 开始训练（不用 Trainer） """


def train_without_trainer():
    """ 一、设定超参数 """
    max_epoch = 959
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0

    """ 二、读入数据，生成模型和优化器 """
    x, t = spiral.load_data()
    model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
    optimizer = SGD(lr=learning_rate)

    # 学习过的变量
    data_size = len(x)
    max_iters = data_size // batch_size
    total_loss = 0
    loss_count = 0
    loss_list = []

    for epoch in range(max_epoch):
        # 打乱数据，不然拟合度会差
        idx = np.random.permutation(data_size)
        x = x[idx]
        t = t[idx]

        # 以 batch_size 分割数据
        for iters in range(max_iters):
            batch_x = x[iters * batch_size:(iters + 1) * batch_size]
            batch_t = t[iters * batch_size:(iters + 1) * batch_size]

        # 计算梯度，更新参数（也就是每次的最优梯度）
        loss = model.forward(batch_x, batch_t)
        model.backward()
        # 模型反向传播计算过后，相应的最优方向存在了 params 和 grads 变量中
        # 我们模型就根据这些保存好的参数来进行更新
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 打印出学习过程反馈给执行者看
        if (iters + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d | iter %d / %d | loss %.2f' % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0


def train_with_trainer():
    """ 一、设定超参数 """
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0

    """ 二、读入数据，生成模型和优化器 """
    x, t = spiral.load_data()
    model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
    optimizer = SGD(lr=learning_rate)

    trainer = Trainer(model, optimizer)
    trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
    trainer.plot()


train_with_trainer()
