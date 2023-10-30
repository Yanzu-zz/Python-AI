import random
import time

import torch
import torch.nn as nn
from torch import optim
from datasets import readLangs, SOS_token, EOS_token, MAX_LENGTH
from models import EncoderRNN, AttenDecoderRNN
from utils import timeSince

device = ("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = MAX_LENGTH + 1

lang1 = "en"
lang2 = "cn"

# 读取训练数据
path = "./data/en-cn.txt"
input_lang, output_lang, pairs = readLangs(lang1, lang2, path)


# print(pairs)
# print(len(pairs))
# print(input_lang.n_words)
# print(input_lang.index2word)
#
# print(out_lang.n_words)
# print(out_lang.index2word)

def listTotensor(input_lang, data):
    indexes_in = [input_lang.word2index[word] for word in data.split(" ")]
    indexes_in.append(EOS_token)
    input_tensor = torch.tensor(indexes_in, dtype=torch.long).view(-1, 1)

    return input_tensor


def tensorsFromPair(pair):
    input_tensor = listTotensor(input_lang, pair[0])
    output_tensor = listTotensor(output_lang, pair[1])

    return (input_tensor, output_tensor)


# 损失函数
def loss_func(input_tensor, output_tensor,
              encoder, decoder,
              encoder_optimizer, decoder_optimizer):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_len = input_tensor.size(0)
    output_len = output_tensor.size(0)

    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

    for ei in range(input_len):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([SOS_token], device=device)
    decoder_hidden = encoder_hidden

    loss = 0
    # 0.5 概率是否使用上一层输出
    use_teacher_forcing = True if random.random() < 0.5 else False
    if use_teacher_forcing:
        for di in range(output_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs
            )

            loss += criterion(decoder_output, output_tensor[di])
            decoder_input = output_tensor[di]
    else:
        for di in range(output_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs
            )
            loss += criterion(decoder_output, output_tensor[di])
            topV, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            if decoder_input.item() == EOS_token:
                break

    # 梯度传播
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / output_len


# 超参数
hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = AttenDecoderRNN(hidden_size, output_lang.n_words, max_len=MAX_LENGTH, dropout=0.1)
encoder.to(device)
decoder.to(device)

lr = 0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr)

scheduler_encoder = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=1, gamma=0.95)
scheduler_decoder = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=1, gamma=0.95)

# loss
criterion = nn.NLLLoss()
n_iters = 100000
training_pairs = [
    tensorsFromPair(random.choice(pairs)) for i in range(n_iters)
]

print_every = 100
save_every = 1000
print_loss_total = 0
start = time.time()
total_loss = 0
for iter in range(1, n_iters + 1):
    training_pair = training_pairs[iter - 1]
    input_tensor = training_pair[0]
    output_tensor = training_pair[1]

    loss = loss_func(input_tensor, output_tensor,
                     encoder, decoder,
                     encoder_optimizer, decoder_optimizer)

    print_loss_total += loss
    if iter % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print("{}, {}, {}, {}".format(timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

    # 保存模型
    if iter % save_every == 0:
        torch.save(encoder.state_dict(), "./models/encoder_{}.pth".format(iter))
        torch.save(decoder.state_dict(), "./models/decoder_{}.pth".format(iter))

    if iter % 10000 == 0:
        scheduler_encoder.step()
        scheduler_decoder.step()
