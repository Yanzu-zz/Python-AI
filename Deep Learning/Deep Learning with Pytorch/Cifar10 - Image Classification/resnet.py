import torch
import torch.nn as nn
import torch.nn.functional as F


# 跳连结构（残差网络）
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        # 跳连分支（shortcut），要和 layer 输出大小匹配
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out1 = self.layer(x)
        out2 = self.shortcut(x)
        # 因为是残差网络，故是跳上去直接相加的（也就是上面说的大小要匹配上）
        out = out1 + out2
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, ResBlock):
        super(ResNet, self).__init__()
        self.in_channel = 32

        # 加层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU()
        )

        # 注意，一个 layer block 包含 3个 卷积层，并且有跳连操作
        # self.layer1 = ResBlock(in_channel=32, out_channel=64, stride=2)
        # self.layer2 = ResBlock(in_channel=64, out_channel=64, stride=2)
        # self.layer3 = ResBlock(in_channel=128, out_channel=128, stride=2)

        # 考虑到 ResNet 有很深的层，故我们写个 for 循环来搭建 layer block
        # 这里先定义两个block练习，标准是 ResNet50，也就是 num_block=50
        self.layer1 = self.make_layer(ResBlock, out_channel=64, stride=1, num_blocks=2)
        self.layer2 = self.make_layer(ResBlock, out_channel=128, stride=2, num_blocks=2)
        self.layer3 = self.make_layer(ResBlock, out_channel=256, stride=2, num_blocks=2)
        self.layer4 = self.make_layer(ResBlock, out_channel=512, stride=2, num_blocks=2)

        # 全连接层，打平数据
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        # print(out.shape)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # 就是映射到 10 个类别的概率分布上去
        # out = F.softmax(out)
        return out

    def make_layer(self, block, out_channel, stride, num_blocks):
        # 存放层的列表
        layers_list = []
        for i in range(num_blocks):
            in_stride = stride if i == 0 else 1
            layers_list.append(block(self.in_channel, out_channel, in_stride))
            self.in_channel = out_channel
        # layers_list = [block(self.in_channel, out_channel, stride if i == 0 else 1) for i in range(num_block)]

        # 展开串联 num_block 个网络
        return nn.Sequential(*layers_list)

        # strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        # layers = []
        # for stride in strides:
        #     layers.append(block(self.in_channel, out_channel, stride))
        #     self.in_channel = out_channel
        # return nn.Sequential(*layers)


def resnet():
    return ResNet(ResBlock)
