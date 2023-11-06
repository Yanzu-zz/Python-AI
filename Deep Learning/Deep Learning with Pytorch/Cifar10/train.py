import os
import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet
from load_cifar10 import train_loader, test_loader

if __name__ == '__main__':
    # 判断是否能用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch_num = 2
    lr = 0.01
    batch_size = 128
    net = VGGNet()
    net = net.to(device)

    # loss
    loss_func = nn.CrossEntropyLoss()

    # optimizer
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # 指数衰减的方法调整学习率（每5轮学习率变为上轮的 0.9x）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # train
    for epoch in range(epoch_num):
        # train BatchNorm, dropout
        net.train()

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_func(outputs, labels)

            # 老三样（更新参数）
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, pred = torch.max(outputs.data, dim=1)
            # correct = pred.eq(labels.data).cpu().sum()

            # print("epoch is: ", epoch)
            # print("train lr is ", optimizer.state_dict()["param_groups"][0]["lr"])
            # print("step: ", i, " loss is: ", loss.item())

        # 测试
        sum_loss = 0
        sum_correct = 0
        for i, data in enumerate(test_loader):
            net.eval()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = loss_func(outputs, labels)

            # 计算准确率
            _, pred = torch.max(outputs.data, dim=1)
            correct = pred.eq(labels.data).cpu().sum()
            sum_loss += loss.item()
            sum_correct += correct.item()

        test_loss = sum_loss * 1.0 / len(test_loader)
        test_correct = sum_correct * 100.0 / len(test_loader) / batch_size
        print("epoch is: ", epoch + 1, "loss is: ", test_loss, "test correct is: ", test_correct)

        # 保存模型
        # if not os.path.exists("./models"):
        #     os.mkdir("./models")
        # torch.save(net.state_dict(), "./models/{}.pth".format(epoch + 1))
        scheduler.step()
