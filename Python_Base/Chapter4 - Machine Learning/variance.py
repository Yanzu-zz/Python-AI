import numpy as np

## Data (rows: stocks / cols: stock prices)
X = np.array([[25, 27, 29, 30],
              [1, 5, 3, 2],
              [12, 11, 8, 3],
              [1, 1, 2, 2],
              [2, 6, 2, 2]])

min_row = min([(i, np.var(X[i])) for i in range(len(X))], key=lambda x: x[1])

print("Row with minimum variance: " + str(min_row[0]))
print("Variance: " + str(min_row[1]))

# second method
var = np.var(X, axis=1)
min_row2 = (np.where(var == min(var)), min(var))
print(var)
print(var == min(var))
print(min_row2)
