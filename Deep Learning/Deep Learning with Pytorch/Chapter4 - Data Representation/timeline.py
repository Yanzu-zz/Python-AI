import numpy as np
import torch

bilkes_numpy = np.loadtxt('../data/p1ch4/bike-sharing-dataset/hour-fixed.csv',
                          dtype=np.float32,
                          delimiter=',',
                          skiprows=1,
                          # 将日期字符串转换为与第1列中的月和日对应的数字
                          converters={1: lambda x: float(x[8:10])}
                          )

bikes = torch.from_numpy(bilkes_numpy)
print(bikes)
