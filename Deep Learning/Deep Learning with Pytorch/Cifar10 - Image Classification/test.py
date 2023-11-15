# 测试模型

import torch
import glob
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from base_resnet import resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = resnet()
# 加载之前训练好的模型参数
net.load_state_dict(torch.load(""))

im_list = glob.glob("./TEST/horse/*")
np.random.shuffle(im_list)
# net.to(device)

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

test_transform = transforms.Compose([
    transforms.CenterCrop((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 逐个读取图片并使用模型预测一遍
for im_path in im_list:
    net.eval()
    im_data = Image.open(im_path)

    inputs = test_transform(im_data)
    # inputs=inputs.to(device)
    inputs = torch.unsqueeze(inputs, dim=0)
    outputs = net.forward(inputs)

    # 获取结果
    _, pred = torch.max(outputs.data, dim=1)
    print(label_name[pred.cpu().numpy()[0]])

    # 查看该图片
    img = np.asarray((im_data))
    # 改变 RGB以及通道 维度顺序
    img = img[:, :, [1, 2, 0]]
    img = cv2.resize(img, (300, 300))
    cv2.imshow("im", img)
    cv2.waitKey()
