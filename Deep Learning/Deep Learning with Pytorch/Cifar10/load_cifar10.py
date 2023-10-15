import glob

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

label_name = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

label_dict = {name: idx for idx, name in enumerate(label_name)}


def default_loader(path):
    return Image.open(path).convert("RGB")


train_transform = transforms.Compose([
    transforms.RandomCrop(28),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])


# train_transform = transforms.Compose([
#     # 裁剪一下
#     transforms.RandomResizedCrop((28, 28)),
#     # 随机翻转
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(90),
#     # 转换成灰度
#     transforms.RandomGrayscale(0.1),
#     # 颜色增强
#     transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
#     # 转换成张量
#     transforms.ToTensor()
# ])


# 自定义数据加载类
class MyDataset(Dataset):
    # im_list 是所有图片的路径
    def __init__(self, im_list, transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = [[im_item, label_dict[im_item.split("\\")[-2]]] for im_item in im_list]
        # for im_item in im_list:
        #     # 路径倒数第二个字符串是该张图片的 label
        #     im_label_name = im_item.split("/")[-2]
        #     imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        # 图片路径和标签
        im_path, im_label = self.imgs[index]
        # 图片数据
        im_data = self.loader(im_path)

        # 若数据有增强
        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data, im_label

    def __len__(self):
        return len(self.imgs)


im_train_list = glob.glob(".\\TRAIN\\*\\*.png")
im_test_list = glob.glob(".\\TEST\\*\\*.png")
# print(im_train_list)

# 获取 训练/测试 图片数据
train_dataset = MyDataset(im_train_list, transform=train_transform)
test_dataset = MyDataset(im_test_list, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=4)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=128,
                         shuffle=False,
                         num_workers=4)

# print("num of train: ", len(train_dataset))
# print("num of test: ", len(test_dataset))
