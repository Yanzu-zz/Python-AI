import csv
import numpy as np
import torch

wine_path = '../data/p1ch4/tabular-wine/winequality-white.csv'
wineq_numpy = np.loadtxt(wine_path, dtype=np.float32, delimiter=';', skiprows=1)
print(wineq_numpy)

col_list = next(csv.reader(open(wine_path), delimiter=';'))
print(wineq_numpy.shape)
print(col_list)

wineq = torch.from_numpy(wineq_numpy)
print(wineq.shape)
print(wineq.dtype)

data = wineq[:, :-1]

target = wineq[:, -1].long()
target_onehot = torch.zeros(target.shape[0], 10)
print(target_onehot.scatter_(1, target.unsqueeze(1), 1.0))

target_unsqueezed = target.unsqueeze(1)
data_mean = torch.mean(data, dim=0)
data_var = torch.var(data, dim=0)
data_normalized = (data - data_mean) / torch.sqrt(data_var)

# 找出坏酒
bad_data = data[target <= 3]
mid_data = data[((target > 3) & (target < 7))]
good_data = data[target >= 7]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

total_sulfur_threshold = 141.83
total_sulfur_data = data[:, 6]
# larger than >
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)

actual_indexes = target > 5
n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
