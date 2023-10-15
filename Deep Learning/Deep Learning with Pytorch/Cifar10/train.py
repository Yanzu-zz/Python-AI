import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet
from load_cifar10 import train_loader, test_loader

if __name__ == '__main__':
    # 判断是否能用 GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_num = 200
    lr = 0.01
    net = VGGNet()
    # net = net.to(device)

    # loss
    loss_func = nn.CrossEntropyLoss()

    # optimizer
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # 指数衰减的方法调整学习率（每5轮学习率变为上轮的 0.9x）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # train
    for epoch in range(epoch_num):
        print("epoch is: ", epoch)
        # train BatchNorm, dropout
        net.train()

        for i, data in enumerate(train_loader):
            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_func(outputs, labels)

            # 老三样（更新参数）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("step: ", i, " loss is: ", loss.item())
