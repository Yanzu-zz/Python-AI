import numpy as np


# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


t1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# 均方值越小越好
print(mean_squared_error(np.array(y1), np.array(t1)))
print(mean_squared_error(np.array(y2), np.array(t1)))

# 越接近0越好
print(cross_entropy_error(np.array(y1), np.array(t1)))
print(cross_entropy_error(np.array(y2), np.array(t1)))
