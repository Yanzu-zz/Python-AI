import torch

# a 可以改变的
a = torch.ones(3)
print(a)
print(a[1])
print(float(a[2]))

a[0] = 6.2
print(a)

b = torch.tensor([1.2, 2.0, 3])

points = torch.tensor([
    [4.0, 1.0],
    [5.0, 3.0],
    [2.0, 1.0]
])
print(points)
print(points.shape)

# 切片操作也一样
print(points[1:])
print(points[0::2])

# 初始化 RGB 三通道张量数据
img_t = torch.randn(3, 5, 5)  # 维度 channels, rows, columns
weights = torch.tensor([0.2126, 0.7152, 0.0722])

# 增加批量次数
batch_t = torch.randn(2, 3, 5, 5)  # 维度 batch, channels, rows, columns

# 求平均值之类的操作
# 因为一个图片张量不一定只有 RGB 通道数据
# 但 RGB 通道数据一定在最后三列，故我们用-3求
img_gray_naive = img_t.mean(-3)
batch_gray_naive = batch_t.mean(-3)
print(img_gray_naive.shape)
print(batch_gray_naive.shape)

unsqueezed_weights = weights.unsqueeze(-1).unsqueeze(-1)
img_weights = (img_t * unsqueezed_weights)
batch_weights = (batch_t * unsqueezed_weights)
img_gray_weighted = img_weights.sum(-3)
batch_gray_weighted = batch_weights.sum(-3)

print(batch_weights.shape)
print(batch_t.shape)
print(unsqueezed_weights.shape)

weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=["channels"])
print(weights_named)

img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')
print("img named: ", img_named.shape, img_named.names)
print("batch named: ", batch_named.shape, batch_named.names)

weights_aligned = weights_named.align_as(img_named)
print(weights_aligned.shape, weights_aligned.names)

gray_named = (img_named * weights_aligned).sum('channels')
print(gray_named.shape, gray_named.names)
gray_plain = gray_named.rename(None)
print(gray_plain.shape, gray_plain.names)
