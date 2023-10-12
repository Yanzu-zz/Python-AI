import torch
import torch.utils.data as data_utils
import torchvision.datasets as dataset
import torchvision.transforms as transforms

# data
# 导入手写识别训练集
train_data = dataset.MNIST(root="mnist", train=True, transform=transforms.ToTensor(), download=True)
test_data = dataset.MNIST(root="mnist", train=False, transform=transforms.ToTensor(), download=False)

# batchsize 也就是每次从数据集中选择一部分进行训练，训练完再到下一批
# 因为网络带宽可能有限，又或者图片数据太大
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=64, shuffle=True)


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


cnn = CNN()
# 把计算从 CPU 转到 GPU 上面
cnn = cnn.cuda()

# loss
# 由于是分类问题，故我们这里采用交叉熵的损失函数
loss_func = torch.nn.CrossEntropyLoss()

# optimizer
# 这里的优化器可以采用 FGD或者Adam 等优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# training
# 对全部样本数据，进行多轮的重复的训练
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        # 把计算从 CPU 转到 GPU 上面
        images = images.cuda()
        labels = labels.cuda()

        # 输出结果
        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        # 进行反向传播来完成对参数的优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch is {}, item is {}/{}, loss i {}\n".format(epoch + 1, i, len(train_data) // 64, loss.item()))

    # eval/test
    loss_test = 0
    accuracy = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = cnn(images)

        # 维度为 [batchsize]
        # output = batchsize * cls_num(10)
        loss_test += loss_func(outputs, labels)
        _, pred = outputs.max(1)

        # 最后统计预测正确的样本数量
        accuracy += (pred == labels).sum().item()

    accuracy = accuracy / len(test_data)
    loss_test = loss_test / (len(test_data) // 64)

    # 最后就是输出查看
    print("epoch is {}, accuracy is {}, "
          "loss test is {}\n".format(epoch + 1, accuracy, loss_test.item()))

torch.save(cnn, "./model/mnist_model.pkl")

# load

# inference
