import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """ 计算 y_true 和 y_predict 之间的准确率"""
    assert y_true.shape[0] == y_predict.shape[0]

    return sum(y_true == y_predict) / len(y_true)


# 线性回归衡量的指标
# 和 scikit-learn 调用方式和函数命名一致
def mean_squared_errot(y_true, y_predict):
    """ 计算 y_true 和 y_predict 之间的 MSE"""
    assert len(y_true) == len(y_predict)

    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    return sqrt(mean_squared_errot(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    assert len(y_true) == len(y_predict)

    return np.sum(np.abosulte(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """ 计算两个参数之间的 R Square """
    return 1 - mean_squared_errot(y_true, y_predict) / np.var(y_true)
