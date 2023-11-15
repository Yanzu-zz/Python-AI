import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

font = FontProperties(fname="C:\\Users\\IAdmin\\AppData\\Local\\Microsoft\\Windows\\Fonts\\DingTalk JinBuTi.ttf")

df = pd.read_csv('./data/housing-data.txt', sep='\s+', header=0)
X = df[['LSTAT']].values
y = df[['MEDV']].values

# 训练模型
# 增加二次、三次放，也就是二次项和三次项回归
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
# 训练二次项和三次项回归得到二次方和三次方的 x
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)
# 增加一些 x轴坐标点，用来 查看训练好的模型
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

lr = LinearRegression()
# 线性回归
lr.fit(X, y)
lr_predict = lr.predict(X_fit)
lr_r2 = r2_score(y, lr.predict(X))

# 二项式回归
lr.fit(X_quad, y)
quad_predict = lr.predict(quadratic.fit_transform(X_fit))
# 计算二项式回归的 R2 值
quadratic_r2 = r2_score(y, lr.predict(X_quad))

# 三项式回归
lr = lr.fit(X_cubic, y)
cubic_predict = lr.predict(cubic.fit_transform(X_fit))
# 计算三项式回归的 R2 值
cubic_r2 = r2_score(y, lr.predict(X_cubic))

print(lr.score(X_cubic, y))
print(cubic_r2)

# 可视化
plt.scatter(X, y, c='gray', edgecolor='white', marker='s', label='训练数据')
plt.plot(X_fit, lr_predict, c='r',
         label='线性(d=1),$R^2={:.2f}$'.format(lr_r2), linestyle='--', lw=3)
plt.plot(X_fit, quad_predict, c='g',
         label='平方(d=2),$R^2={:.2f}$'.format(quadratic_r2), linestyle='-', lw=3)
plt.plot(X_fit, cubic_predict, c='b',
         label='立方(d=3),$R^2={:.2f}$'.format(cubic_r2), linestyle=':', lw=3)
plt.xlabel('地位较低人口的百分比[LSTAT]', fontproperties=font)
plt.ylabel('以1000美元为计价单位的房价[RM]', fontproperties=font)
plt.title('波士顿房价预测', fontproperties=font, fontsize=20)
plt.legend(prop=font)
plt.show()
