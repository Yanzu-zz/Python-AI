import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import matplotlib.pyplot as plt
import seaborn as sns


class BasicNN(nn.Module):
    def __init__(self):
        super(BasicNN, self).__init__()

        # 因为是按照图上已经优化完毕的网络，故我们不需要再进行梯度更新了
        # 也就是 requires_grad=False
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

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


input_doses = torch.linspace(start=0, end=1, steps=11)
print(input_doses)

model = BasicNN()
output_values = model(input_doses)
print(output_values)

# 接着就是可视化
# 先设置个好看的样式
sns.set(style="whitegrid")
# 用线画出来
sns.lineplot(x=input_doses, y=output_values,
             color='green', linewidth=2.5)
plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show()
