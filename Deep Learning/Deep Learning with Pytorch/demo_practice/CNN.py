import torch

# net 人工网络结构
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 采用序列工具来构建网络结构
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        # 因为上面有池化pooling操作，此时我们的图像会变成 14*14 大小，通道数为 32
        # 因为是识别 0-9 共10个数字，所以第二个参数是10维的
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)

    # 前向传播
    def forward(self, x):
        out = self.conv(x)
        # conv后是个4阶Tensor，我们将它拉成一个向量
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
