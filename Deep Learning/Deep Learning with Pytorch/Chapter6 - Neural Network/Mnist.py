import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

# data
train_data = dataset.MNIST(root='./mnist',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)
test_data = dataset.MNIST(root='./mnist',
                          train=False,
                          transform=transforms.ToTensor(),
                          download=False)

# 因为数据可能会很庞大，故我们每次就之选 batch_size 个数据进行训练
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64,
                                     shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data,
                                    batch_size=64,
                                    shuffle=True)


# net
class CNN(torch.nn.Module):
    def __init__(self):
        # 继承父类
        super(CNN, self).__init__()

        # 加层
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        # 因为有池化层，conv过后大小除二变成 14x14
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)

    def forward(self, x):
        out = self.conv(x)
        # 拉成一维向量
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


cnn = CNN()
# cnn = cnn.cuda()

# loss
loss_func = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# training
for epoch in range(1):
    for i, (images, labels) in enumerate(train_loader):
        # images = images.cuda()
        # # 正确的答案
        # labels = labels.cuda()
        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch is {}, ite is {}/{}， loss is {}"
              .format(epoch + 1, i, len(train_data) // 64, loss.item()))

    # eval/test
    loss_test = 0
    accuracy = 0
    for i, (images, labels) in enumerate(train_loader):
        outputs = cnn(images)
        # [batchsize]
        # outputs = batchsize*10
        loss_test += loss_func(outputs, labels)
        # 拿到最大的那个结果
        _, pred = outputs.max(1)
        accuracy = (pred == labels).sum().item()

    accuracy = accuracy / len(test_data)
    loss_test = loss_test / (len(test_data) // 64)

    print("epoch is {}, accuracy is {}, loss_test is {}"
          .format(epoch + 1, accuracy, loss_test.item()))

# save model
torch.save(cnn, "./model/mnist_model.pkl")
