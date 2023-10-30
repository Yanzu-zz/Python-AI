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

# 加载模型（参数）
model_text_cls.load_state_dict(torch.load("./models/100.pth"))

for i, batch in enumerate(train_dataloader):
    label, data = batch
    label = torch.tensor(label, dtype=torch.int64)
    data = torch.tensor(data)
    # label.to(cfg.devices)
    # data.to(cfg.devices)

    # 拿到输出结果
    pred_softmax = model_text_cls.forward(data)
    # 拿到预测值
    pred = torch.argmax(pred_softmax, dim=1)
    # print(pred_softmax)
    # print(label)
    # print(pred)

    out = torch.eq(pred, label)
    accuracy = out.sum() * 1.0 / pred.size()[0]
    print(accuracy)
