import numpy as np
import sys, os

sys.path.append(os.pardir)
from common.layers import MatMul
from common.util import preprocess, convert_one_hot


# 实现 CBOW 模型的推理# 实现 CBOW 模型的推理# 实现 CBOW 模型的推理
# 注意 CBOW 模型是没有激活函数的！（Sigmoid，Relu 等）
def cbow_predict():
    # 样本的上下文数据
    c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
    c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

    # 权重的初始值
    W_in = np.random.randn(7, 3)
    W_out = np.random.randn(3, 7)

    # 下面的结构就是按照书上的 CBOW 结构图创建的
    # 生成层（2个 in，1个 out
    in_layer0 = MatMul(W_in)
    in_layer1 = MatMul(W_in)
    out_layer = MatMul(W_out)

    # 开始走神经网络计算（正向传播）
    h0 = in_layer0.forward(c0)
    h1 = in_layer1.forward(c1)
    # 书上写的就是 前后两个单词节点上下文 的权值的平均值
    h = (h0 + h1) / 2
    # 最后就是走下输出层
    # s 就是结果，我们可以应用 softmax 函数（加多下一层）将结果转成 one-hot 概率矩阵
    s = out_layer.forward(h)
    print(s)


# corpus 是词库
def create_contexts_target(corpus, window_size=1):
    # 以 window_size 为左右边界，一个一个元素分割数组
    target = corpus[window_size:-window_size]

    # 下面来生成 contexts 数组（每个单词的前后 window_size 个单词）
    # 也就是从左遍历到右 2*window_size 个元素（不包括自身）加进 contexts
    contexts = []
    # 每个单词都遍历
    for idx in range(window_size, len(corpus) - window_size):
        tmp = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            tmp.append(corpus[idx + t])
        contexts.append(tmp)

    return np.array(contexts), np.array(target)


# 简易使用一下
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus, window_size=1)
print('The original form:')
# print(contexts)
# print(target)

# 接着就是转换为 one-hot 矩阵
vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)
print('The one-hot structure: ')
print(contexts)
# print(target)
