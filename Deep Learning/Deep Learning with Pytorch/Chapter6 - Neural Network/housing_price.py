import re
import torch
import numpy as np
import torch.optim as optim

# data
# data = np.loadtxt("./housing.data", dtype=np.float32, delimiter=" ", skiprows=0)
# ff = open("./housing.data").readlines()
# for item in ff:
#     out = re.sub(r"\s{2,}", " ", item).strip()
#     print(out)

# one-liner
data = np.array([(re.sub(r"\s{2,}", " ", item).strip()).split(" ") for item in open("./housing.data")]) \
    .astype(np.float16)
print(data.shape)

X = data[:, :-1]
Y = data[:, -1]

X_train = X[:-10, ...]
Y_train = Y[:-10, ...]
X_test = X[-10:, ...]
Y_test = Y[-10:, ...]
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# net
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()

        # 隐藏层
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_output)

    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out


net = Net(13, 1)

# loss
loss_func = torch.nn.MSELoss()

# optimizer
optimizer = optim.Adam(net.parameters(), lr=0.01)

# training
epoch = 3000
for i in range(epoch):
    x_data = torch.tensor(X_train, dtype=torch.float32)
    y_data = torch.tensor(Y_train, dtype=torch.float32)

    pred = net.forward(x_data)
    # 删除一个维度，使得和 y 一样
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data) * 0.001

    # 调用优化器
    # 梯度置为 0
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("ite: {}, loss_train: {}".format(i, loss))
    print(pred[:10])
    print(y_data[:10])

    # test
    x_data = torch.tensor(X_test, dtype=torch.float32)
    y_data = torch.tensor(Y_test, dtype=torch.float32)

    pred = net.forward(x_data)
    # 删除一个维度，使得和 y 一样
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, y_data) * 0.001
    print("ite: {}, loss_test: {}".format(i, loss_test))

# 保存整个模型
torch.save(net, './model/housing_price_model.pkl')
# 只保存参数
# torch.save(net.state_dict(), './model/houseing_price_params.pkl')