import imageio.v2 as imageio
import torch
import os

img_arr = imageio.imread('../data/p1ch4/image-dog/bobby.jpg')
# 这里是宽，高，通道，不是我们想要的格式，下面可以改变它
print(img_arr.shape)
# print(img_arr)

img = torch.from_numpy(img_arr)
# 改变布局
out = img.permute(2, 0, 1)
print(out.shape)

batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

# 加载一个目录中的所有图片
data_dir = '../data/p1ch4/image-cats/'
filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.png']

for i, filename in enumerate(filenames):
    img_arr = imageio.imread(os.path.join(data_dir, filename))
    img_t = torch.from_numpy(img_arr)
    # 变成 通道数，宽，高
    img_t = img_t.permute(2, 0, 1)
    # 去除可能多的维度
    img_t = img_t[:3]
    batch[i] = img_t

# 正规化数据
n_channels = batch.shape[1]
for c in range(n_channels):
    mean = torch.mean(batch[:, c])
    std = torch.std(batch[:, c])
    batch[:, c] = (batch[:c] - mean) / std
