import torch
import torch.nn as nn
from torch import optim
from models import Model
from datasets import data_loader, textCls
from configs import Config

cfg = Config()

data_path = "./sources/weibo_senti_100k.csv"
data_stop_path = "./sources/hit_stopword"
dict_path = "./sources/dict"
dataset = textCls(dict_path, data_path, data_stop_path)
train_dataloader = data_loader(dataset, cfg)

# model
cfg.pad_size = dataset.max_len_seq
model_text_cls = Model(config=cfg)
# model_text_cls.to(cfg.devices)

# loss
loss_func = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model_text_cls.parameters(), lr=cfg.learn_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# training step
for epoch in range(cfg.num_epochs):
    for i, batch in enumerate(train_dataloader):
        label, data = batch
        label = torch.tensor(label, dtype=torch.int64)
        data = torch.tensor(data)
        # label.to(cfg.devices)
        # data.to(cfg.devices)

        # 梯度置为0
        optimizer.zero_grad()
        # 拿到输出结果
        pred = model_text_cls.forward(data)
        # 计算 loss
        loss_val = loss_func(pred, label)
        # print(pred)
        # print(label)

        print("epoch is {}, ite is {}, val is {}"
              .format(epoch, i, loss_val))

        # 进行反向传播
        loss_val.backward()

        # 对参数进行更新
        optimizer.step()

        # torch.save(model_text_cls.state_dict(), "./models/{}.pth".format(i + 1))

    scheduler.step()

# 保存模型
torch.save(model_text_cls.state_dict(), "./models/{}.pth".format(cfg.num_epochs))
