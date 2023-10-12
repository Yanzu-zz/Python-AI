# 用我们在 demo_regression 文件中保存好的模型使用
# 也就是不用再 fit 了
import re
import torch
import numpy as np

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

# 也就是不用再训练模型
# 上面数据定义是不能省的，也就是相当于新的数据用保存好的模型来预测
net = torch.load("./model/model.pkl")
loss_func = torch.nn.MSELoss()

x_data = torch.tensor(X_test, dtype=torch.float32)
y_data = torch.tensor(Y_test, dtype=torch.float32)
pred = net.forward(x_data)
pred = torch.squeeze(pred)
loss_test = loss_func(pred, y_data) * 0.001
print("loss_test:{}".format(loss_test))
