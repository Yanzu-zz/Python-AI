import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

font = FontProperties(fname="C:\\Users\\IAdmin\\AppData\\Local\\Microsoft\\Windows\\Fonts\\DingTalk JinBuTi.ttf")

df = pd.read_csv('./data/housing-data.txt', sep='\s+', header=0)
X = df[['LSTAT']].values
y = df[['MEDV']].values

# np.log() 默认以 e 为底数
# 假设 X和y 有对数关系
y_sqrt = np.log(y)

# 训练模型
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
lr = LinearRegression()

# 线性回归
lr.fit(X, y)
lr_predict = lr.predict(X_fit)
# 计算线性回归的 R2 值
lr_r2 = r2_score(y, lr.predict(X))

plt.scatter(X, y, c='gray', edgecolors='white', marker='s', label='训练数据')
plt.plot(X_fit, lr_predict, c='r', label='线性,%R^2={:.2f}$'.format(lr_r2))
plt.xlabel('地位较低人口的百分比[LSTAT]', fontproperties=font)
plt.ylabel('ln(以1000美元为计价单位的房价[RM])', fontproperties=font)
plt.title('波士顿房价预测', fontproperties=font, fontsize=20)
plt.legend(prop=font)
plt.show()
