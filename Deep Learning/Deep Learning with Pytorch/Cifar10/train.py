import os

import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet
from resnet import resnet
from mobilenetv1 import mobilenetv1_small
from inceptionMoldule import InceptionNetSmall
from load_cifar10 import train_loader, test_loader
import tensorboardX

if __name__ == '__main__':
    # 判断是否能用 GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_num = 1
    lr = 0.01
    batch_size = 128
    # net = VGGNet()
    # net = resnet()
    # net = mobilenetv1_small()
    net = InceptionNetSmall()
    # net = net.to(device)

    # loss
    loss_func = nn.CrossEntropyLoss()

    # optimizer
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 指数衰减的方法调整学习率（每5轮学习率变为上轮的 0.9x）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # 日志
    if not os.path.exists("./log"):
        os.mkdir("./log")
    writer = tensorboardX.SummaryWriter("log")
    step_n = 0

    # train
    for epoch in range(epoch_num):
        # print("epoch is: ", epoch)
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

            _, pred = torch.max(outputs.data, dim=1)
            correct = pred.eq(labels.data).cpu().sum()

            im = torchvision.utils.make_grid(inputs)

            writer.add_scalar("train loss", loss.item(), global_step=step_n)
            writer.add_scalar("train correct", 100.0 * correct.item() / batch_size, global_step=step_n)
            writer.add_image("train im", im, global_step=step_n)

            print(i)

            step_n += 1

        # 对学习率进行更新
        scheduler.step()

        sum_loss = 0
        sum_correct = 0
        for i, data in enumerate(test_loader):
            net = net.eval()
            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            _, pred = torch.max(outputs.data, dim=1)
            correct = pred.eq(labels.data).cpu().sum()

            sum_loss += loss
            sum_correct += correct

            im = torchvision.utils.make_grid(inputs)
            writer.add_scalar("test loss", loss.item(), global_step=epoch + 1)
            writer.add_scalar("test correct", 100.0 * correct.item() / batch_size, global_step=epoch + 1)
            writer.add_image("test im", im, global_step=step_n)

            print(i)

        print("epoch: ", epoch + 1,
              " loss: ", sum_loss * 1.0 / len(test_loader),
              " mini-batch correct: ", sum_correct * 100.0 / len(test_loader) / batch_size)

    if not os.path.exists("./models"):
        os.mkdir("./models")
    torch.save(net.state_dict(), "./models/{}.pth".format("vggnet"))

    writer.close()
