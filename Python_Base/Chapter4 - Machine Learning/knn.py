import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Data (House Size (square meters) / House Price ($))
X = np.array([[35, 30000], [45, 45000], [40, 50000],
              [35, 35000], [25, 32500], [40, 40000]])

print(X[:, 0])
print(X[:, 0].reshape((len(X), -1)))
print(X[:, 0].reshape((-1, 1)))

# 注意，这是分类任务，不可能预测某个不存在的房屋面积对应的价格，只能是分类到给定的数据值
KNN_Cls = KNeighborsClassifier(n_neighbors=3).fit(X[:, 0].reshape((len(X), -1)), X[:, 1])
# 这才是回归，能预测不在训练数据上的点的对应价格（注意，此时只是简单的将 3 个邻居的价格相加然后求平均值）
KNN_Reg = KNeighborsRegressor(n_neighbors=3).fit(X[:, 0].reshape((len(X), -1)), X[:, 1])

# y_predict = np.array([x for x in range(20, 30)])
# y_predict = y_predict.reshape((len(y_predict), -1))
# print(KNN.predict(y_predict))

y_predict = np.array([x for x in range(20, 40)])
y_predict = y_predict.reshape((len(y_predict), -1))
print(KNN_Cls.predict(y_predict))
print(KNN_Reg.predict(y_predict))
