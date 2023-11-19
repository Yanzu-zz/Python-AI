import os.path

import tensorboardX
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PIL import Image
import torch
from models import Discriminator, Generator
from utils import *
from datasets import ImageDataset
import itertools

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batchsize = 1
    size = 256  # 图片尺寸
    data_root = "./datasets/apple2orange"

    netG_A2B = Generator().to(device)
    netG_B2A = Generator().to(device)


    netG_A2B.load_state_dict(torch.load("./models/netG_A2B.pth"))
    netG_B2A.load_state_dict(torch.load("./models/netG_B2A.pth"))

    netG_A2B.eval()
    netG_B2A.eval()

    input_A = torch.ones([1, 3, size, size], dtype=torch.float)
    input_B = torch.ones([1, 3, size, size], dtype=torch.float)

    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    dataloader = DataLoader(ImageDataset(data_root, transforms_, model="test"), batch_size=batchsize, shuffle=False,
                            num_workers=16)

    if not os.path.exists("./outputs/A"):
        os.mkdir("./outputs/A")
    if not os.path.exists("./outputs/B"):
        os.mkdir("./outputs/B")

    # 对数据进行 inference
    for i, batch in enumerate(dataloader):
        real_A = torch.tensor(input_A.copy_(batch['A']), dtype=torch.float)
        real_B = torch.tensor(input_B.copy_(batch['B']), dtype=torch.float)

        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        # 保存模型转换后的结果（苹果 to 橘子，vice versa
        save_image(fake_A, "./outputs/A/{}.png".format(i))
        save_image(fake_B, "./outputs/B/{}.png".format(i))
        print(i)
