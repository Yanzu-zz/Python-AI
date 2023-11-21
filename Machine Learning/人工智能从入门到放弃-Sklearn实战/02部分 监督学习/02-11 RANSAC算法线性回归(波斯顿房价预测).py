import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import RANSACRegressor, LinearRegression

font = FontProperties(fname="C:\\Users\\IAdmin\\AppData\\Local\\Microsoft\\Windows\\Fonts\\DingTalk JinBuTi.ttf")

# 读取数据
df = pd.read_csv('./data/housing-data.txt', sep='\s+', header=0)
X = df[['RM']].values
y = df[['MEDV']].values

# 训练模型
# max_trials=88即最大迭代次数为88次
# min_samples=66即样本最低数量为66个
# loss=‘absolute_loss’即使用均方误差损失函数
# residual_threshold=6即只允许与拟合线垂直距离在6个单位以内的采样点被包括在内点集
# 可以看到定义其实和线性回归什么一样，只不过需要传的参数多了点
ransac = RANSACRegressor(LinearRegression(),
                         max_trials=88,
                         min_samples=66,
                         loss='absolute_error',
                         residual_threshold=6)
ransac.fit(X, y)

# 获取内点集（这是索引）
inlier_mask = ransac.inlier_mask_
# 非内点集（这也是索引）
outlier_mask = np.logical_not(inlier_mask)
# 建立回归线
line_X = np.arange(3, 10, 1)
# 由于ransac模型期望数据存储在二维阵列中，因此使用line_X[:, np.newaxis]方法给X增加一个新维度
line_y_ransac = ransac.predict(line_X[:, np.newaxis])

# 可视化
# 内外点（也就是数据点离不离群）
plt.scatter(X[inlier_mask], y[inlier_mask], c='r',
            edgecolors='white', marker='s', label='内点')
plt.scatter(X[outlier_mask], y[outlier_mask], c='g',
            edgecolors='white', marker='o', label='离群点')
plt.plot(line_X, line_y_ransac, color='k')

plt.xlabel('平均房间数目[MEDV]', fontproperties=font)
plt.ylabel('以1000美元为计价单位的房价[RM]', fontproperties=font)
plt.title('波士顿房价预测', fontproperties=font, fontsize=20)
plt.legend(prop=font)
plt.show()
print('RANSAC算法线性回归斜率:{}'.format(ransac.estimator_.coef_[0]))
