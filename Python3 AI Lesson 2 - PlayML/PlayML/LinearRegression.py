import numpy as np
from metrics import r2_score


# 广义上的 线性回归模型，即支持单元和多元线性回归
class LinearRegression:
    def __init__(self):
        # coefficient 系数
        self.coef_ = None
        # 截距
        self.intercept_ = None
        # θ，是私有变量
        self._theta = None

    # 使用推导出的数学公式进行训练（下面有速度更快的方法）
    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0]

        # 第一列全是0，剩下都是 X_i
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        # 正规军数学公式，套用就行
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    # 梯度下降法训练线性回归模型
    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """ 根据训练数据集，使用梯度下降法训练 Linear Regression 模型"""
        assert X_train.shape[0] == y_train.shape[0]

        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            # res = np.empty(len(theta))
            # res[0] = np.sum(X_b.dot(theta) - y)
            # for i in range(1, len(theta)):
            #     res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            # return res * 2 / len(X_b)

            # 使用推导出来的向量化公式优化
            return (X_b.T.dot(X_b.dot(theta) - y)) * 2. / len(X_b)

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

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

    # 随机梯度下降法函数
    def fit_sgd(self, X_train, y_train, n_iters=5, t0=5, t1=50):
        """ 根据训练数据集，使用梯度下降法训练 Linear Regression 模型 """
        assert X_train.shape[0] == y_train.shape[0]
        assert n_iters >= 1

        def learning_rate(t):
            return t0 / (t + t1)

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta, n_iters, t0=5, t1=50):
            theta = initial_theta
            m = len(X_b)
            for cur_iter in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(cur_iter * m + i) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

    def predict(self, X_predict):
        assert self.intercept_ is not None and self.coef_ is not None
        assert X_predict.shape[1] == len(self.coef_)

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """ 根据测试数据集确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"
