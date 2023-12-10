import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns


class BasicLightningTrain(L.LightningModule):
    def __init__(self):
        super(BasicLightningTrain, self).__init__()

        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.learning_rate = 0.01

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

        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias
        output = F.relu(input_to_final_relu)
        return output

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = (output_i - label_i) ** 2

        return loss


inputs = torch.tensor([0., 0.5, 1.])
labels = torch.tensor([0., 1., 0.])
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

model = BasicLightningTrain()
# 训练器，可以让它自行（并行地）跑到多个 GPUs 上
trainer = L.Trainer(max_epochs=34, accelerator="auto", devices="auto")
tuner = L.pytorch.tuner.Tuner(trainer)
# 使用训练器找到合适的 学习率
# 也就是 lr.find() 会生成 100 个学习率（0.001-1.0)，查看谁最优
lr_find_results = tuner.lr_find(model=model,
                                train_dataloaders=dataloader,
                                min_lr=0.001,
                                max_lr=1.0,
                                early_stop_threshold=None)
# 拿到优化学习率
new_lr = lr_find_results.suggestion()
print(f"lr_find() suggests {new_lr:.5f} for the learning rate.")
model.learning_rate = new_lr

trainer.fit(model, train_dataloaders=dataloader)
print(model.final_bias)

input_doses = torch.linspace(start=0, end=1, steps=11)

output_values = model(input_doses)
# 先设置个好看的样式
sns.set(style="whitegrid")
# 用线画出来
sns.lineplot(x=input_doses, y=output_values.detach(),
             color='green', linewidth=2.5)
plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show()
