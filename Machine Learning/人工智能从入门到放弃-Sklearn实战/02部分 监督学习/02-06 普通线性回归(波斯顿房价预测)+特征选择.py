import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 基于matplotlib的Python可视化库。 它提供了一个高级界面来绘制有吸引力的统计图形
import seaborn as sns
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression

font = FontProperties(fname="C:\\Users\\IAdmin\\AppData\\Local\\Microsoft\\Windows\\Fonts\\DingTalk JinBuTi.ttf")

# 读取数据
df = pd.read_csv('./data/housing-data.txt', sep='\s+', header=0)
# print(df.head())

# 特征选择（选择三列特征，可以自由选择，但前提是知道对应的列是什么数据）
cols = ['RM', 'MEDV', 'LSTAT']
# 构造散列特征之间的联系即构造散点图矩阵
# sns.pairplot(df[cols], height=3)
# plt.tight_layout()
# plt.show()

# 求解上述散列特征的相关系数
'''
对于一般的矩阵X，执行A=corrcoef(X)后，A中每个值的所在行a和列b，反应的是原矩阵X中相应的第a个列向量和第b个列向量的相似程度（即相关系数）
'''
# cm = np.corrcoef(df[cols].values.T)
# # 控制颜色刻度即颜色深浅
# sns.set(font_scale=2)
# # 构造关联矩阵
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',
#                  annot_kws={'size': 20}, yticklabels=cols, xticklabels=cols)
# plt.show()

# 训练模型
X = df[['RM']].values
y = df[['MEDV']].values
lr = LinearRegression()
lr.fit(X, y)

plt.scatter(X, y, color='r', s=30, edgecolor='white', label='训练数据')
plt.plot(X, lr.predict(X), c='g')
plt.xlabel('平均房间数目[MEDV]', fontproperties=font)
plt.xlabel('以1000美元为计价单位的房价[RM]', fontproperties=font)
plt.title('波士顿房价预测', fontproperties=font, fontsize=20)
# legend 就是给图像加个图例
plt.legend(prop=font)
plt.show()
print('普通线性回归斜率: {}'.format(lr.coef_[0]))
