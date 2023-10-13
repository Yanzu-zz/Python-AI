import numpy as np
from sklearn.linear_model import LinearRegression

apple = np.array([155, 156, 157])
n = len(apple)

print(np.arange(n))
# 生成 x=0,1,2，与 apple 数据对应一下给模型训练
print(np.arange(n).reshape(n, -1))
Model = LinearRegression().fit(np.arange(n).reshape((n, -1)), apple)

print(Model.predict([[3]]))
print(Model.predict([[5], [8]]))
