import numpy as np


def accuracy_score(y_true, y_predict):
    """ 计算 y_true 和 y_predict 之间的准确率"""
    assert y_true.shape[0] == y_predict.shape[0]

    return sum(y_true == y_predict) / len(y_true)
