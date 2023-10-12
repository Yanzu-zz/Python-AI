import torch

a = torch.Tensor([[1, 2], [3, 4]])
print(a)
print(a.type())

# 全1
b = torch.ones(2, 3)
print(b)

# 全0
c = torch.zeros(2, 3)
print(c)

# 和 a的shape一样，但元素都是零
# ones_like 同理
d = torch.zeros_like(a)
print(d)

# 随机类
# 下面 rand 的参数2,2是指定 shape，没有指定随机范围（默认0-1）
a = torch.rand(2, 2)
print(a)

# 正态分布
b = torch.normal(mean=0.0, std=torch.rand(5))
print(b)

# 均匀分布
c = torch.Tensor(2, 2).uniform_(-1, 1)
print(c)

# 定义序列
# arange 是左闭右开，也就是 [0,10)，且这里步长为 1
a = torch.arange(0, 10, 1)
print(a)

# 等间隔生成序列，其中长度为 4
b = torch.linspace(2, 10, 4)
print(b)

# numpy 也有类似的生成函数
import numpy as np

a = np.array([[1, 2], [2, 3]])
print(a)
