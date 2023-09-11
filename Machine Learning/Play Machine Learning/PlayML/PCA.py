import numpy as np


class PCA:
    def __init__(self, n_components):
        """ 初始化 PCA """
        assert n_components >= 1

        self.n_components = n_components
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        """ 根据训练数据集 X 获得数据的均值和方差 """
        assert self.n_components <= X.shape[1]

        def demean(X):
            return X - np.mean(X, axis=0)

        # 笔记中的 f 函数
        # X 需要先进行 demean
        def f(w, X):
            return np.sum((X.dot(w) ** 2)) / len(X)

        # 推导出的向量化公式
        def df(w, X):
            return X.T.dot(X.dot(w)) * 2. / len(X)

        # 转换 w 为单位向量
        def direction(w):
            return w / np.linalg.norm(w)

        # 梯度上升法
        def first_component(X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
            cur_iter = 0
            w = direction(initial_w)

            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                # 这里是加号，这就是梯度上升法与梯度下降法不同的地方
                w = w + eta * gradient
                # 每次求一个单位方向
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break
                cur_iter += 1

            return w

        # 开始求前 n 个主成分
        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta, n_iters)
            self.components_[i, :] = w

            # 减去上个主成分的量，用剩下的继续下一轮PCA
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    # 高维数据向低维数据进行映射
    def transform(self, X):
        """ 将给定的X，映射到各个主成分分量中 """
        assert X.shape[1] == self.components_.shape[1]

        # 具体看推出来的公式
        return X.dot(self.components_.T)

    # 低维数据恢复成高维数据，注意，此时有信息永久丢失了
    def inverse_transform(self, X):
        """ 将非定的X，反向映射回原来的特征空间 """
        assert X.shape[1] == self.components_.shape[0]

        # 也是看回公式
        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
