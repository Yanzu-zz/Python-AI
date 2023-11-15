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
plt.scatter(x1, y1, s=50, color='b', label="male(+1)")
plt.scatter(x2, y2, s=50, color='r', label="female(-1)")
plt.vlines(2.8, 0, 2, colors='r', linestyles="-", label='%ws+b=0%')
plt.title("线性可分", fontsize=20)
plt.xlabel('x')

# 一行二列第二个
# 这个是演示男孩中的一个去到了女生区域，无法用线性方程区分出两个阵营的情况
plt.subplot(122)
plt.scatter(x1, y1, s=50, color='b', label="male(+1)")
plt.scatter(x2, y2, s=50, color='r', label="female(-1)")
plt.scatter(3.5, 1, s=50, color='b')
plt.vlines(2.8, 0, 2, colors='r', linestyles="-", label='%ws+b=0%')
plt.title("线性可分", fontsize=20)
plt.xlabel('x')

plt.legend(loc='upper right')
plt.show()
