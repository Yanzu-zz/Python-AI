import numpy as np

# Data: air quality index AQI data (row = city)
X = np.array(
    [[42, 40, 41, 43, 44, 43],  # Hong Kong
     [30, 31, 29, 29, 29, 30],  # New York
     [8, 13, 31, 11, 11, 9],  # Berlin
     [11, 11, 12, 13, 11, 12]])  # Montreal

cities = np.array(['Hong Kong', 'New York', 'Berlin', 'Montreal'])

# 这里会将平均值（一个数）广播成 X.shape 形状的 np array
print(X > np.average(X))
print((X > np.average(X))[1])
print(np.nonzero(X > np.average(X)))
print(np.nonzero(X > np.average(X))[0])
print(cities[np.nonzero(X > np.average(X))[0]])
print(set(np.nonzero(X > np.average(X))[0]))

# 找到污染程度大于平均值的城市
polluted = set(cities[np.nonzero(X > np.average(X))[0]])
print(polluted)
