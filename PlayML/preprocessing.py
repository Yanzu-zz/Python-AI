import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    # 这样写能和 scikit-learn 的函数风格保持一致
    def fit(self, X):
        """ 根据训练数据集 X 获得数据的均值和方差 """
        assert X.ndim == 2, "只处理二维数据"

        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])

        return self

    def transform(self, X):
        """ 将 X 根据这个 StandardScaler 进行均值方差归一化处理 """
        assert X.ndim == 2
        assert self.mean_ is not None and self.scale_ is not None
        assert X.shape[1] == len(self.mean_)

        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]

        return resX
