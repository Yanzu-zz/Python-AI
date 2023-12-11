import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions import Uniform
from torch.utils.data import TensorDataset, DataLoader

import lightning as L
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


class WordEmbeddingFromScratch(L.LightningModule):
    def __init__(self):
        super(WordEmbeddingFromScratch, self).__init__()

        # weights 初始化的范围
        min_value = -0.5
        max_value = 0.5

        self.input1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.input4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.output1_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output1_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output2_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output3_w2 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w1 = nn.Parameter(Uniform(min_value, max_value).sample())
        self.output4_w2 = nn.Parameter(Uniform(min_value, max_value).sample())

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        input = input[0]

        inputs_to_top_hidden = ((input[0] * self.input1_w1) +
                                (input[1] * self.input2_w1) +
                                (input[2] * self.input3_w1) +
                                (input[3] * self.input4_w1))
        inputs_to_bottom_hidden = ((input[0] * self.input1_w2) +
                                   (input[1] * self.input2_w2) +
                                   (input[2] * self.input3_w2) +
                                   (input[3] * self.input4_w2))

        output1 = ((inputs_to_top_hidden * self.output1_w1) + (inputs_to_bottom_hidden * self.output1_w2))
        output2 = ((inputs_to_top_hidden * self.output2_w1) + (inputs_to_bottom_hidden * self.output2_w2))
        output3 = ((inputs_to_top_hidden * self.output3_w1) + (inputs_to_bottom_hidden * self.output3_w2))
        output4 = ((inputs_to_top_hidden * self.output4_w1) + (inputs_to_bottom_hidden * self.output4_w2))
        # 因为上面定义的交叉熵loss会自动帮我们执行 softmax 函数，这里就堆叠起来就行
        output_presoftmax = torch.stack([output1, output2, output3, output4])
        return (output_presoftmax)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        # 计算损失
        loss = self.loss(output_i, label_i[0])
        return loss

    def visiualizeData(self):
        data = {
            "w1": [
                self.input1_w1.item(),
                self.input2_w1.item(),
                self.input3_w1.item(),
                self.input4_w1.item(),
            ],
            "w2": [
                self.input1_w2.item(),
                self.input2_w2.item(),
                self.input3_w2.item(),
                self.input4_w2.item(),
            ],
            "token": ["Troll2", "is", "great", "Gymkata"],
            "input": ["input1", "input2", "input3", "input4"]
        }

        df = pd.DataFrame(data)
        print(df)

        # 可视化
        sns.scatterplot(data=df, x="w1", y="w2")
        plt.text(df.w1[0], df.w2[0], df.token[0],
                 horizontalalignment='left',
                 size='medium',
                 color='black',
                 weight='semibold')

        plt.text(df.w1[1], df.w2[1], df.token[1],
                 horizontalalignment='left',
                 size='medium',
                 color='black',
                 weight='semibold')
        plt.text(df.w1[2], df.w2[2], df.token[2],
                 horizontalalignment='left',
                 size='medium',
                 color='black',
                 weight='semibold')
        plt.text(df.w1[3], df.w2[3], df.token[3],
                 horizontalalignment='left',
                 size='medium',
                 color='black',
                 weight='semibold')
        plt.show()


# training data
inputs = torch.tensor([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
])

labels = torch.tensor([
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
    [0., 1., 0., 0.],
])

# 合并它们
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)
modelFromScratch = WordEmbeddingFromScratch()

print("Before optimization, the parameters are...")
modelFromScratch.visiualizeData()

trainer = L.Trainer(max_epochs=100)
trainer.fit(modelFromScratch, train_dataloaders=dataloader)

print("After training: ")
modelFromScratch.visiualizeData()

softmax = nn.Softmax(dim=0)
print(torch.round(softmax(modelFromScratch(torch.tensor([[1., 0., 0., 0.]]))), decimals=2))
print(torch.round(softmax(modelFromScratch(torch.tensor([[0., 1., 0., 0.]]))), decimals=2))
print(torch.round(softmax(modelFromScratch(torch.tensor([[0., 0., 1., 0.]]))), decimals=2))
print(torch.round(softmax(modelFromScratch(torch.tensor([[0., 0., 0., 1.]]))), decimals=2))
