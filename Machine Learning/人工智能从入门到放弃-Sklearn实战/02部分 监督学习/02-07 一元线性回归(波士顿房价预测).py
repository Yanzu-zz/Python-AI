import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

font = FontProperties(fname="C:\\Users\\IAdmin\\AppData\\Local\\Microsoft\\Windows\\Fonts\\DingTalk JinBuTi.ttf")

# 获取数据
df = pd.read_csv('./data/housing-data.txt', sep='\s+', header=0)
X = df.iloc[:, :-1].values
y = df['MEDV'].values

# 训练测试数据分离
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 训练模型
lr = LinearRegression()
lr.fit(X_train, y_train)
# 预测训练集和测试集数据
y_train_predict = lr.predict(X_train)
y_test_predict = lr.predict(X_test)

# 可视化
# train predict 和 真实y_train 数据的误差
plt.scatter(y_train_predict, y_train_predict - y_train, c='r',
            marker='s', edgecolors='white', label='训练数据')
# test predict 和 真实y_test 数据的误差
plt.scatter(y_test_predict, y_test_predict - y_test, c='g',
            marker='o', edgecolors='white', label='测试数据')
plt.xlabel('预测值', fontproperties=font)
plt.ylabel('误差值', fontproperties=font)
# 可视化 y=0 的一条直线也就是误差为0的直线
plt.hlines(y=0, xmin=-1, xmax=50, color='k')
plt.xlim(-10, 50)
plt.legend(prop=font)
plt.show()

# 训练集的均方误差
train_mse = mean_squared_error(y_train, y_train_predict)
# 测试集的均方误差
test_mse = mean_squared_error(y_test, y_test_predict)
print('训练集的均方误差: {}'.format(train_mse))
print('测试集的均方误差: {}'.format(test_mse))
