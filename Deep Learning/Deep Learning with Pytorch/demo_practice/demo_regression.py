import re  # 正则
import torch
import numpy as np

# 读取数据文件 fetch data
# 因为原始数据文件分割列时空格数量不一致，所以我们可以用正则表达式来解决
ff = open("../Chapter6 - Neural Network/housing.data").readlines()
data = []
for item in ff:
    # 去掉换行，统一空格格式
    out = re.sub(r"\s{2,}", " ", item).strip()
    print(out)
    # 然后加入数组中
    data.append(out.split(" "))
# 转成需要的格式
data = np.array(data).astype(float)
print(data.shape)

# 除了最后一列是 Y，其它都是 X
X = data[:, 0:-1]
Y = data[:, -1]

# 分割训练和测试数据集
X_train = X[0:496, ...]
Y_train = Y[0:496, ...]
X_test = X[496, ...]
Y_test = Y[496, ...]


# 搭建人工神经网络 net
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        # 建立一个线性回归模型
        # 定义了只有一个隐藏层的简单神经网络
        # 当然还可以再加几个隐藏层
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_output)

    # 前向传播
    def forward(self, x):
        out = self.hidden(x)
        # 加入非线性的运算
        out = torch.relu(out)
        out = self.predict(out)
        return out


net = Net(13, 1)  # 13 为特征的数量，1 为输出数量

# 定义损失函数 loss
loss_func = torch.nn.MSELoss()

# 定义优化器 optomiter
# optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
# 可以使用不同的优化函数
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# 训练模型 training
# 训练1000次
for i in range(10000):
    x_data = torch.tensor(X_train, dtype=torch.float32)
    y_data = torch.tensor(Y_train, dtype=torch.float32)
    # 然后就是前向运算
    pred = net.forward(x_data)
    # 弄成维度一致再计算下面的 loss
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data) * 0.001

    # 优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("ite:{}, loss_train:{}".format(i, loss))
    print(pred[0:10])
    print(y_data[0:10])

    # 测试
    x_data = torch.tensor(X_test, dtype=torch.float32)
    y_data = torch.tensor(Y_test, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, y_data) * 0.001
    print("ite:{}, loss_test:{}".format(i, loss_test))

# 保存模型，也就是保存我们训练好的模型
# 下面这种保存方式我们下次想用的时候直接 torch.load 就行
torch.save(net, "./model/model.pkl")
# 而这种保存方式需要先定义出来，再 torch.load 加载
torch.save(net.state_dict(), "./model/params.pkl")
