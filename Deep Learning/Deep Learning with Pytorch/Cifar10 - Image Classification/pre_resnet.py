import torch.nn as nn
from torchvision import models


# 用 pytorch 定义好的模型进行运算
class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.num_feature = self.model.fc.in_features
        # 设置输出成我们想要的
        self.model.fc = nn.Linear(self.num_feature, 10)

    def forward(self, x):
        out = self.model(x)
        return out


def pytorch_resnet18():
    return resnet18()
