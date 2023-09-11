import numpy as np
from metrics import r2_score


class SimpleLinearRegression1:

    def __init__(self):
        """ 初始化 Simple Linear Regression 模型 """
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """ 根据训练数据集训练模型 """
        # 也就是使用 最小二乘法 来计算 a和b
        assert x_train.ndim == 1
        assert len(x_train) == len(y_train)

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """ 给定待预测数据集 x_predict，返回表示 x_predict 的结果向量 """
        assert x_predict.ndim == 1
        assert self.a_ is not None and self.b_ is not None

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """ 给定单个但预测数据 x_single，返回 x_single 的预测结果 """
        # 也就是用到自己 fit 出的拟合函数，预测给定 x 坐标的值
        return self.a_ * x_single + self.b_

    def __repr(self):
        return "SimpleLinearRegression1()"


# 将 a 向量化运算，加速计算时间
# 这才是主要使用的函数，上面那个就是学习用的
class SimpleLinearRegression:

    def __init__(self):
        """ 初始化 Simple Linear Regression 模型 """
        self.a_ = None
        self.b_ = None

    # 相比于 1， 只有 fit 函数变了
    def fit(self, x_train, y_train):
        """ 根据训练数据集训练模型 """
        # 也就是使用 最小二乘法 来计算 a和b
        assert x_train.ndim == 1
        assert len(x_train) == len(y_train)

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # 不使用 for 循环，用 numpy 的向量化计算方法
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """ 给定待预测数据集 x_predict，返回表示 x_predict 的结果向量 """
        assert x_predict.ndim == 1
        assert self.a_ is not None and self.b_ is not None

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """ 给定单个但预测数据 x_single，返回 x_single 的预测结果 """
        # 也就是用到自己 fit 出的拟合函数，预测给定 x 坐标的值
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        """ 根据测试数据集确定当前模型的精确度 """
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr(self):
        return "SimpleLinearRegression1()"
