import numpy as np

# Data: yearly salary in ($1000) [2017, 2018, 2019]
alice = [99, 101, 103]
bob = [110, 108, 105]
tim = [90, 88, 85]

names = np.array(['Alice', 'Bob', 'Tim'])
salaries = np.array([alice, bob, tim])
taxation = np.array([[0.2, 0.25, 0.22],
                     [0.4, 0.5, 0.5],
                     [0.1, 0.2, 0.1]])

# print(salaries * taxation)

# 注意，这里的 salaries*taxation 不是矩阵乘法，而是 哈达玛积Hadamard Product
# 与就是对应位置的元素直接相乘即可
max_income = np.max(salaries - salaries * taxation)
# print(max_income)

tmp = salaries * taxation
print(tmp >= np.max(tmp))
print(np.nonzero(tmp >= np.max(tmp)))

# [0][0] 是取出行的第一个元素（这里只有一个）
max_income_info = np.nonzero(tmp >= np.max(tmp))[0][0]
print(names[max_income_info])
print(salaries[max_income_info])
print(taxation[max_income_info])
