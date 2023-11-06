# 感知机引入图例
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

# %matplotlib inline

np.random.seed(1)
# 生成两个性别的数据
x1 = np.random.random(20) + 1.5
y1 = np.random.random(20) + 0.5
x2 = np.random.random(20) + 3
y2 = np.random.random(20) + 0.5

# 一行二列第一个
plt.subplot(121)
plt.scatter(x1, y1, s=50, color='b', label="男孩")
plt.scatter(x2, y2, s=50, color='b', label="女孩")


