import numpy as np

# Data: popular Instagram accounts (millions followers)
# 注意，因为 inst 包含数字和字符串，numpy 自动将 inst 类型转为 非数字 类型
# 故如想要将第一列元素与数字比大小，需要先转换类型
inst = np.array([[232, "@instagram"],
                 [133, "@selenagomez"],
                 [59, "@victoriassecret"],
                 [120, "@cristiano"],
                 [111, "@beyonce"],
                 [76, "@nike"]])

print(inst[:, 0].astype(np.float16) > 100)
print((inst[inst[:, 0].astype(np.float16) > 100])[:, 1])
print((inst[inst[:, 0].astype(np.float16) > 100, 1]))

# 只要第一列（也就是粉丝数）> 100 号才加入结果数组
superstars = inst[inst[:, 0].astype(np.float16) > 100, 1]
print(superstars)
