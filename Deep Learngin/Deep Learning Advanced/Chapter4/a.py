import numpy as np

# 从 0-9 的数字中随机选择一个数字`
print(np.random.choice(10))
print(np.random.choice(10))

words = ['you', 'say', 'goodbye', 'I', 'Hello', '.']

# 从 words 列表中随机选择一个元素
print(np.random.choice(words))

# 有放回采样5次
print(np.random.choice(words, size=5))

# 无放回采样5次
print(np.random.choice(words, size=5, replace=False))

# 基于概率分布进行采样
p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
print(np.random.choice(words, p=p))

# 令低频数据概率变高点
p2 = [0.7, 0.1, 0.1, 0.01, 0.09]
new_p2 = np.power(p2, 0.75)
new_p2 /= np.sum(new_p2)
print(new_p2)
