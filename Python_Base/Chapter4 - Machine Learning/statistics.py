## Dependencies
import numpy as np

## Stock Price Data: 5 companies
# (row=[price_day_1, price_day_2, ...])
X = np.array([[8, 9, 11, 12],
              [1, 2, 2, 1],
              [2, 8, 9, 9],
              [9, 6, 6, 3],
              [3, 3, 3, 3]])

avg, var, std = np.average(X, axis=1), np.var(X, axis=1), np.std(X, axis=1)
print(avg)
print(var)
print(std)

