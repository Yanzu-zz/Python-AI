import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import MAX_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 编码器
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        # 加层
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        # 拿到学习后的结果
        output, hidden = self.gru(output, hidden)

        return output, hidden

    # 初始化第一个隐藏层
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 解码器
# 这里不带 attention 的结构
# 后面会加上，以达到对比学习
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()

        # 加层
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        # 打平，线性化拿到输出
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))

        return output, hidden

    # 初始化第一个隐藏层
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 基于 Attention 解码 RNN 方式
class AttenDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1, max_len=MAX_LENGTH):
        super(AttenDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout
        self.max_len = max_len

        # 加层
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # 这里就是加了 Attention 层
        # 利用网络拿到 max_len 长度的权重值
        self.attn = nn.Linear(self.hidden_size * 2, self.max_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    # 拼接组件
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        atten_weight = F.softmax(
            self.attn(torch.cat([embedded[0], hidden[0]], 1)),
            dim=1
        )

        att_applied = torch.bmm(
            atten_weight.unsqueeze(0),
            encoder_outputs.unsqueeze(0),
        )

        output = torch.cat([embedded[0], att_applied[0]], dim=1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, atten_weight

    # 初始化第一个隐藏层
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 简单测试
if __name__ == "__main__":
    encoder_net = EncoderRNN(5000, 256)
    decoder_net = DecoderRNN(256, 5000)
    atten_decoder_net = AttenDecoderRNN(256, 5000)

    tensor_in = torch.tensor([12, 14, 16, 18], dtype=torch.long).view(-1, 1)
    hidden_in = torch.zeros(1, 1, 256)
    encoder_out, encoder_hidden = encoder_net(tensor_in[0], hidden_in)

    print(encoder_out)
    print(encoder_hidden)

    tensor_in = torch.tensor([100]).view(-1, 1)
    hidden_in = torch.zeros(1, 1, 256)
    encoder_out = torch.zeros(10, 256)

    out1, out2, out3 = atten_decoder_net(tensor_in, hidden_in, encoder_out)
    print(out1)
    print(out2)
    print(out3)

    out1, out2 = decoder_net(tensor_in, hidden_in)
    print(out1)
    print(out2)
