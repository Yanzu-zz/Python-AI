from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([
    [0, "No"],
    [10, "No"],
    [27, "No"],
    [30, "Yes"],
    [60, "Yes"],
    [90, "Yes"]
])

print(X[:, 0])
# 因为 sklearn 的函数都是需要 Array[ [Array1],[Array2] ] 这种格式，所以还需要 reshape 一下
# 而 对应的正确输出 y，则不需要 二维格式，直接一维输入即可
print(X[:, 0].reshape(len(X), -1))

model = LogisticRegression().fit(X[:, 0].reshape((len(X), -1)), X[:, 1])

y_predict = np.array([2, 12, 13, 20, 33, 45, 40, 44, 46, 50, 99])
y_predict = y_predict.reshape((len(y_predict), -1))
print(model.predict(y_predict))

for i in range(40):
    print("x=" + str(i) + " --> " + str(model.predict_proba([[i]])))
