from CNN import CNN
import numpy as np
import cv2
import torch
import torch.utils.data as data_utils
import torchvision.datasets as dataset
import torchvision.transforms as transforms

# data
# 导入手写识别训练集
test_data = dataset.MNIST(root="mnist", train=False, transform=transforms.ToTensor(), download=False)

# batchsize 也就是每次从数据集中选择一部分进行训练，训练完再到下一批
# 因为网络带宽可能有限，又或者图片数据太大
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=64, shuffle=True)

# cnn = CNN()
cnn = torch.load("./model/mnist_model.pkl")
# 把计算从 CPU 转到 GPU 上面
cnn = cnn.cuda()

# loss
# 由于是分类问题，故我们这里采用交叉熵的损失函数
loss_func = torch.nn.CrossEntropyLoss()

# optimizer
# 这里的优化器可以采用 FGD或者Adam 等优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

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

    # 将预测结果画图出来
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()
    # 此时 batchsize = 1*28*28
    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_label = labels[idx]
        im_pred = pred[idx]
        im_data = im_data.transpose(1, 2, 0)

        print("label", im_label)
        print("pred", im_pred)
        cv2.imshow("imdata", im_data)
        cv2.waitKey(0)


accuracy = accuracy / len(test_data)
loss_test = loss_test / (len(test_data) // 64)
# 最后就是输出查看
print("accuracy is {}, "
      "loss test is {}\n".format(accuracy, loss_test.item()))
