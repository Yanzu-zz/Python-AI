import numpy as np
from metrics import accuracy_score


class LogisticRegression:
    def __init__(self):
        """ 初始化模型 """
        # coefficient 系数
        self.coef_ = None
        # 截距
        self.interception_ = None
        # θ，是私有变量
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    # 梯度下降法训练逻辑回归模型
    # 因为逻辑回归没有数学公式解，只能用梯度下降法来求解
    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """ 根据训练数据集，使用梯度下降法训练 Logistic Regression 模型"""
        assert X_train.shape[0] == y_train.shape[0]

        # 损失函数
        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            # 使用推导出来的向量化公式优化
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)

        # 优化梯度下降函数，防止死循环
        # n_iters：最多循环次数
        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta)

        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]

    def predict_proba(self, X_predict):
        """ 给定待预测数据集，返回表示 X_predict 的结果概论向量 """
        assert self.interception_ is not None and self.coef_ is not None
        assert X_predict.shape[1] == len(self.coef_)

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        # 将结果数值转化为概论
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        """ 给定待预测数据集，返回表示 X_predict 的结果概论向量 """
        assert self.interception_ is not None and self.coef_ is not None
        assert X_predict.shape[1] == len(self.coef_)

        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """ 根据测试数据集确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        # 逻辑回归判断的是分类准确度
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"
