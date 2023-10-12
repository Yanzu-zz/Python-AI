# 用训练好的模型进行推理
import re
import torch
import numpy as np


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

net = torch.load('./model/housing_price_model.pkl')
data = np.array([(re.sub(r"\s{2,}", " ", item).strip()).split(" ") for item in open("./housing.data")]) \
    .astype(np.float16)

X = data[:, :-1]
Y = data[:, -1]

X_train = X[:-10, ...]
Y_train = Y[:-10, ...]
X_test = X[-10:, ...]
Y_test = Y[-10:, ...]

loss_func = torch.nn.MSELoss()

# test
x_data = torch.tensor(X_test, dtype=torch.float32)
y_data = torch.tensor(Y_test, dtype=torch.float32)

pred = net.forward(x_data)
# 删除一个维度，使得和 y 一样
pred = torch.squeeze(pred)
loss_test = loss_func(pred, y_data) * 0.001
print("loss_test: {}".format(loss_test))
