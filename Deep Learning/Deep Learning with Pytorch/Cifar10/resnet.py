import torch
import torch.nn as nn
import torch.nn.functional as F


# 跳连结构（残差网络）
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()

        self.layer=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, x):


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

    def forward(self, x):
