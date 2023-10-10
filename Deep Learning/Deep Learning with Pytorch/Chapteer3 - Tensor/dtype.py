import torch

double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
print(double_points.dtype)
print(short_points.dtype)

# 还可以这样转换
double_points2 = torch.zeros(10, 2).double()
short_points2 = torch.zeros(3, 6).short()
print(double_points2.dtype)
print(short_points2.dtype)

# 这样更方便（主要是 to 函数能接受其它参数）
double_points3 = torch.zeros(10, 2).to(torch.double)
short_points3 = torch.zeros(3, 6).to(dtype=torch.short)
print(double_points3.dtype)
print(short_points3.dtype)

# 随机生成
# rand 生成 0-1 的值
points_64 = torch.rand(5, dtype=torch.double)
# 但 short 会舍弃小数值，变成0
points_short = points_64.to(dtype=torch.short)
print(points_64 * points_short)
