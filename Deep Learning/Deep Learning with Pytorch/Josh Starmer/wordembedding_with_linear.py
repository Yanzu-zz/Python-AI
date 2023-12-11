import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions import Uniform
from torch.utils.data import TensorDataset, DataLoader

import lightning as L
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


class WordEmbeddingWithLinear(L.LightningModule):
    def __init__(self):
        super(WordEmbeddingWithLinear, self).__init__()

        self.input_to_hidden = nn.Linear(in_features=4, out_features=2, bias=False)
        self.hidden_to_output = nn.Linear(in_features=2, out_features=4, bias=False)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        hidden = self.input_to_hidden(input)
        output_values = self.hidden_to_output(hidden)

        return output_values

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = self.loss(output_i, label_i)

        return loss

    def visiualizeData(self):
        data = {
            "w1": model.input_to_hidden.weight.detach()[0].numpy(),
            "w2": model.input_to_hidden.weight.detach()[1].numpy(),
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
model = WordEmbeddingWithLinear()

print("Before optimization, the parameters are...")
model.visiualizeData()

trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_dataloaders=dataloader)

print("After training: ")
model.visiualizeData()

softmax = nn.Softmax(dim=0)
print(torch.round(softmax(model(torch.tensor([[1., 0., 0., 0.]]))), decimals=2))
print(torch.round(softmax(model(torch.tensor([[0., 1., 0., 0.]]))), decimals=2))
print(torch.round(softmax(model(torch.tensor([[0., 0., 1., 0.]]))), decimals=2))
print(torch.round(softmax(model(torch.tensor([[0., 0., 0., 1.]]))), decimals=2))

# nn.Embedding 来使用这些预训练好的 embedding 值
# 我们发现每个词的两个embedding值是一列，但每次它输出一行，我们只需要转置一下即可
print(model.input_to_hidden.weight)

word_embeddings = nn.Embedding.from_pretrained(model.input_to_hidden.weight.T)
print(word_embeddings.weight)

# 用字典创建映射更方便
# 这样就可以在 transformer 之类的模型里面使用预训练好的 embedding 了
vocab = {
    'Troll 2': 0,
    'is': 1,
    'great': 2,
    'Gymkata': 3
}
print(word_embeddings(torch.tensor(vocab['Troll2'])))
