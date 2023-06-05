import numpy as np
from .metrics import r2_score


# 广义上的 线性回归模型，即支持单元和多元线性回归
class LinearRegression:
    def __init__(self):
        # coefficient 系数
        self.coef_ = None
        # 截距
        self.interception_ = None
        # θ，是私有变量
        self._theta = None

    # 使用推导出的数学公式进行训练（下面有速度更快的方法）
    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0]

        # 第一列全是0，剩下都是 X_i
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        # 正规军数学公式，套用就行
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        assert self.interception_ is not None and self.coef_ is not None
        assert X_predict.shape[1] == len(self.coef_)

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """ 根据测试数据集确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
