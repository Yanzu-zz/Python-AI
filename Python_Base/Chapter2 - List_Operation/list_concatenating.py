import matplotlib.pyplot as plt

cardiac_cycle = [62, 60, 62, 64, 68, 77, 80, 76, 71, 66, 61, 60, 62]

# 去掉开头一个，结尾两个重复元素
# 接着重复10次
cardiac_cycle = cardiac_cycle[1:-2] * 10

plt.plot(cardiac_cycle)
plt.show()
