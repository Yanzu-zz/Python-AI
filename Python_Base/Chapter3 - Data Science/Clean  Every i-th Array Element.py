import numpy as np

# daily stock prices
# [morning, midday, evening]
solar_x = np.array(
    [[1, 2, 3],  # today
     [2, 2, 5]])  # yesterday

# calcuate the average price(today and yesterday)
# axis = 0 就是列（竖着方向）
print(np.average(solar_x, axis=0))
# axis = 1 就是行（恒着方向）
print(np.average(solar_x, axis=1))

## Sensor data (Mo, Tu, We, Th, Fr, Sa, Su)
# 注意，此时 tmp 是个一维数组而已
tmp = np.array([1, 2, 3, 4, 3, 4, 4,
                5, 3, 3, 4, 3, 4, 6,
                6, 5, 5, 5, 4, 5, 5])

print(tmp)
print(tmp[6::7])
print(tmp.reshape((-1, 7)))
print(tmp.reshape((-1, 7))[:, 0:-1])
# axis = 1，计算每行的平均值
print(np.average(tmp.reshape((-1, 7)), axis=1))

# 选择每个星期的星期天，算平均值（这里包括星期天）
tmp[6::7] = np.average(tmp.reshape(-1, 7), axis=1)
# 这里不包括星期天
# tmp[6::7] = np.average(tmp.reshape(-1, 7)[:, 0:-1], axis=1)
print(tmp)
