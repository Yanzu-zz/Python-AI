import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import Lasso, Ridge, ElasticNet

font = FontProperties(fname="C:\\Users\\IAdmin\\AppData\\Local\\Microsoft\\Windows\\Fonts\\DingTalk JinBuTi.ttf")

# 获取数据
df = pd.read_csv('./data/housing-data.txt', sep='\s+', header=0)
X = df[['RM']].values
y = df[['MEDV']].values

# Lasso(L1) 正则回归
# 没错，就是简单调用 Lasso 函数就行
lasso = Lasso(alpha=1.0)
lasso.fit(X, y)
lasso_predict = lasso.predict(X)

# Ridge(L2) 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
ridge_predict = ridge.predict(X)

# ElasticNet 弹性网回归
# l1_ratio=0 时等同于 Lasso 回归
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net_predict = elastic_net.predict(X)

# 可视化
plt.scatter(X, y, c='gray', edgecolor='white', marker='s', label='训练数据')
plt.plot(X, lasso_predict, c='r', label='L1正则化', linestyle='--')
plt.plot(X, ridge_predict, c='b', label='L2正则化', linestyle='-')
plt.plot(X, elastic_net_predict, c='g', label='弹性网络', linestyle=':')
plt.xlabel('平均房间数目[MEDV]', fontproperties=font)
plt.ylabel('以1000美元为计价单位的房价[RM]', fontproperties=font)
plt.title('波士顿房价预测', fontproperties=font, fontsize=20)
plt.legend(prop=font)
plt.show()
