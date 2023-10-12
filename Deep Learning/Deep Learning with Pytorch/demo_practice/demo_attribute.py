import torch

# 合理选择计算设备能实现快速计算和最大化利用硬件
dev = torch.device("cpu")
# 有显卡可以用 cuda:0
# dev = torch.device("cuda:0")

# a = torch.tensor([2, 2], device=dev)
a = torch.tensor([2, 2], dtype=torch.float32, device=dev)
print(a)

# 稀疏矩阵
i = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
v = torch.tensor([1, 2, 3, 5])
b = torch.sparse_coo_tensor(i, v, (4, 4), dtype=torch.float32,device=dev)
print(b)

# 转化成稠密矩阵
c = b.to_dense()
print(c)
