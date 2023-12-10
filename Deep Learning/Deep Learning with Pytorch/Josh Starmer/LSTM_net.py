import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

torch.set_float32_matmul_precision('medium')

class LSTMbyHand(L.LightningModule):
    def __init__(self):
        super(LSTMbyHand, self).__init__()
        # 正态分布的两个变量，我们下面的权重参数，就是按照这个正态分布来进行初始化的
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

        # 下面开始初始化权重（4个LSTM块的权重，每个有3个权重参数）
        self.wlr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wlr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        # b 通常都是初始化为 0
        self.blr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wpr1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wpr2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bpr1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wp1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wp2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bp1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.wo1 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.wo2 = nn.Parameter(torch.normal(mean=mean, std=std), requires_grad=True)
        self.bo1 = nn.Parameter(torch.tensor(0.), requires_grad=True)

    # Do the LSTM math
    def lstm_unit(self, input_value, long_memory, short_memory):
        # LSTM unit 最左边决定本次长期记忆能保留多少百分比
        long_remember_percent = torch.sigmoid((short_memory * self.wlr1) + (input_value * self.wlr2) + self.blr1)

        # LSTM unit 第二个块，用来创建本次输出有多少百分比能进入长期记忆
        potential_remember_percent = torch.sigmoid((short_memory * self.wpr1) + (input_value * self.wpr2) + self.bpr1)
        potential_memory = torch.tanh((short_memory * self.wp1) + (input_value * self.wp2) + self.bp1)
        updated_long_memory = ((long_memory * long_remember_percent) + (potential_remember_percent * potential_memory))

        # LSTM unit 第三个块，用来决定这次输入有多少百分比要短期记忆
        output_percent = torch.sigmoid((short_memory * self.wo1) + (input_value * self.wo2) + self.bo1)
        updated_short_memory = torch.tanh(updated_long_memory) * output_percent

        return ([updated_long_memory, updated_short_memory])

    # Make a forward pass through unrolled LSTM
    def forward(self, input):
        long_memory, short_memory = 0, 0
        # 创建 day'i' = input[i]，其中 day'i' 是一个变量
        for i in range(4):
            globals()[f"day{i + 1}"] = input[i]

        # 逐步 flow 过整个 LSTM
        for i in range(1, 5):
            long_memory, short_memory = self.lstm_unit(globals()[f"day{i}"], long_memory, short_memory)

        return short_memory

    # Configure Adam optimizer
    def configure_optimizers(self):
        return Adam(self.parameters())

    # Calculate loss and log training progress
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i) ** 2

        self.log("train_loss", loss)

        if label_i == 0:
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss


model = LSTMbyHand()

# training data
inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
labels = torch.tensor([0., 1.])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

trainer = L.Trainer(max_epochs=2000, accelerator="auto", devices="auto")
trainer.fit(model, train_dataloaders=dataloader)
path_to_best_checkpoint = trainer.checkpoint_callback.best_model_path
trainer = L.Trainer(max_epochs=5000, accelerator="auto", devices="auto")
trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_best_checkpoint)

print("Now let's compare the observed and predicted value...")
print("Company A: Observed = 0, Predicted = ",
      model(torch.tensor([0., 0.5, 0.25, 1.])).detach())

print("\nNow let's compare the observed and predicted value...")
print("Company B: Observed = 1, Predicted = ",
      model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
