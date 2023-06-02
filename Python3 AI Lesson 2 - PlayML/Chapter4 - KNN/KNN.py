import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score


# 初始用单个函数模拟
# def KNN_classify(k, X_train, y_train, x):
#     assert 1 <= k <= X_train.shape[0]
#     assert X_train.shape[0] == y_train.shape[0]
#     assert X_train.shape[1] == x.shape[0]
#
#     distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in X_train]
#     nearest = np.argsort(distances)
#
#     topK_y = [y_train[i] for i in nearest[:k]]
#     votes = Counter(topK_y)
#
#     return votes.most_common(1)[0][0]


# 自行模拟 scikit-learn 中 KNN 模型的封装
class KNNClassifier:
    def __inti__(self, k):
        assert k >= 1
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """ 根据训练数据集 X_train和y_train 训练KNN分类器 """
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """ 给定待预测数据集 X_predict，返回表示 X_predict 的结果向量 """
        assert self._X_train is not None and self._y_train is not None
        assert X_predict.shape[1] == self._X_train.shape[1]

        y_predict = [self._predict(x) for x in X_predict]
        return y_predict

    def _predict(self, x):
        """ 给定单个但预测数据x，返回x的预测结果值 """
        assert x.shape[0] == self._X_train.shape[1]

        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """ 根据测试数据集传入的参数确定当前模型的准确度 """
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k
