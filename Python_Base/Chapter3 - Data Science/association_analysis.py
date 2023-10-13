import numpy as np

# Data: row is customer shopping basket
# row = [course 1, course 2, ebook 1, ebook 2]
# value 1 indicates that an item was bought.
basket = np.array([[0, 1, 1, 0],
                   [0, 0, 0, 1],
                   [1, 1, 0, 0],
                   [0, 1, 1, 1],
                   [1, 1, 1, 0],
                   [0, 1, 1, 0],
                   [1, 1, 0, 1],
                   [1, 1, 1, 1]])
n = len(basket[0])

print(basket.shape)
print(basket[:, 2:])
print(np.all(basket[:, 2:], axis=1))
print(np.sum(np.all(basket[:, 2:], axis=1)))

# 查看 同时购买两本电子书的顾客 占的百分比
copurchases = np.sum(np.all(basket[:, 2:], axis=1)) / basket.shape[0]
print(copurchases)

# 寻找 最经常被一个顾客买的两件物品，并找出百分比最高的那两个物品
# 注意，这里的 i,j 都是列，也就是两个不同列对比，这里用 : 就可以选择所有行
copurchases2 = [(i, j, np.sum(basket[:, i] + basket[:, j] == 2)) for i in range(n) for j in range(i + 1, n)]
print(copurchases2)

# 接着再输出购买频次最高的那两个物品
print(max(copurchases2, key=lambda x: x[2]))

# 当然我们也可以从头到尾 one-liner 输出
copurchases3 = max(([(i, j, np.sum(basket[:, i] + basket[:, j] == 2)) for i in range(n) for j in range(i + 1, n)]),
                   key=lambda x: x[2])
print(copurchases3)

# 如果是要 top-k 个两个物件，可以排序再 slicing（或者建堆）
copurchases4 = sorted(([(i, j, np.sum(basket[:, i] + basket[:, j] == 2)) for i in range(n) for j in range(i + 1, n)]),
                   key=lambda x: x[2])[:-3:-1]
print(copurchases4)
