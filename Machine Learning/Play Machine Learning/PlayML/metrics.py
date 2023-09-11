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


def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 0))


def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict == 1))


def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 0))


def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict == 1))


def confusion_matrix(y_true, y_predict):
    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)],
    ])


# 有了四个指标的数量，我们就能计算各种率了
def precision_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)

    # 防止除于0
    try:
        return tp / (tp + fp)
    except:
        return 0.0


def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)

    # 防止除于0
    try:
        return tp / (tp + fn)
    except:
        return 0.0


def f1_score(y_true, y_predict):
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    try:
        return 2 * precision * recall / (precision + recall)
    except:
        return 0.0


def TPR(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)

    try:
        return tp / (tp + fn)
    except:
        return 0.0


def FPR(y_true, y_predict):
    fp = FP(y_true, y_predict)
    tn = TN(y_true, y_predict)

    try:
        return fp / (fp + tn)
    except:
        return 0.0
