import torch

a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
print(a, a.shape)
print(a_t, a_t.shape)

points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# 本质是一个大小为6的一维数组（连续数组）
print(points.storage())
print(points.storage()[1])

# 正常编辑也是没问题
points.storage()[0] = 7.4
print(points.storage())

a = torch.ones(3, 2)
print(a)
# 将 a 的所有元素都变成 0
a.zero_()
print(a)

second_point = points[1]
print(second_point)
print(second_point.storage_offset())
print(second_point.size())

print(points.stride())

# 转置
# t 是 transpose 函数的简写
points_t = points.t()
print(points_t)
print(id(points))
print(id(points_t))

print(points.is_contiguous())
print(points_t.is_contiguous())
print(points.contiguous())

# 保存 张量 到本地文件
saved_filename = './points.t'
torch.save(points, saved_filename)
# 用安全方法
with open(saved_filename, 'wb') as f:
    torch.save(points, f)

# 加载也很方便
points2 = torch.load(saved_filename)
# 安全方法
with open(saved_filename, 'rb') as f:
    points2 = torch.load(f)
print(points2)

test1 = torch.tensor(list(range(9)))
print(test1)
