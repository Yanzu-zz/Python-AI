import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import matplotlib.pyplot as plt
import seaborn as sns


# 有参数没有最优化，需要训练
class BasicNN_train(nn.Module):
    def __init__(self):
        super(BasicNN_train, self).__init__()

        # 因为是按照图上已经优化完毕的网络，故我们不需要再进行梯度更新了
        # 也就是 requires_grad=False
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        # 我们这里先假设 final_bias 没有最优化，但 pytorch 的 tensor 有自动梯度优化功能
        # 只需要把 requires_grad=True 即可
        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True)

    # 用 forward 函数来使用上面初始化好的 w和b，连接 input 到 activation function 和 output
    def forward(self, input):
        # 左上第一个 connection
        input_to_top_relu = input * self.w00 + self.b00
        # 连接激活函数（ReLU)
        top_relu_output = F.relu(input_to_top_relu)
        # 再通过第二个连接
        scaled_top_relu_output = top_relu_output * self.w01

        # 左下同理
        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11

        # 接着将它两加起来，送进最后的激活函数，得到网络的输出
        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias
        output = F.relu(input_to_final_relu)
        return output


# 训练数据
inputs = torch.tensor([0., 0.5, 1.])
labels = torch.tensor([0., 1., 0.])
model = BasicNN_train()
optimizer = SGD(model.parameters(), lr=0.1)
print("Final bias, before optimization: " + str(model.final_bias.data) + "\n")

# 训练阶段
# 每次训练完整个训练数据叫做一个 epoch
# 对应这里也就是我们每次遍历完 inputs 的 3 个数据，就称一个 epoch
for epoch in range(100):
    # 每轮训练的 loss，用来查看模型与训练数据的拟合程度
    total_loss = 0

    for iteration in range(len(inputs)):
        input_i = inputs[iteration]
        label_i = labels[iteration]
        output_i = model(input_i)

        # 计算 loss
        loss = (output_i - label_i) ** 2
        # loss.backward() 会累积每次的梯度计算（由 model 保存）
        loss.backward()
        total_loss += float(loss)

    # 查看拟合程度是否小于某个值（自己决定）
    if total_loss < 0.0001:
        print("Num steps: " + str(epoch))
        break

    # 如果拟合程度还不够，我们就用 .step() 来更新 model 里面的参数
    optimizer.step()
    # 然后清空上面 loss.backward() 存储的每轮累加的梯度计算
    # 如果不清空，新一轮 epoch 时候会累加上一轮的梯度数据
    optimizer.zero_grad()

    print(model.final_bias)
    print("Step: " + str(epoch) + " Final Bias: " + str(model.final_bias.data) + "\n")

# 查看训练后的拟合程度
input_doses = torch.linspace(start=0, end=1, steps=11)
print(input_doses)
output_values = model(input_doses)
print(output_values)
# 接着就是可视化
# 先设置个好看的样式
sns.set(style="whitegrid")
# 用线画出来
sns.lineplot(x=input_doses, y=output_values.detach(),
             color='green', linewidth=2.5)
plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show()
