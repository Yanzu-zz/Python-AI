import matplotlib.pyplot as plt
import numpy as np


# import sys,os
# sys.path.append(os.pardir)

# 计算两个向量的余弦相似度（余弦定理）
def cos_similarity(x, y, eps=1e-8):
    # 分母是x和y的L2范数
    # 加个微小值防止除于 0 向量
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)

    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 取出查询词
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 计算余弦相似度
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        # 和各个共线矩阵行计算余弦相似度
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 基于余弦相似度，按降序输出值
    count = 0
    # 按降序排序（索引）
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s ' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


# 语料库预处理
def preprocess(text):
    # text = 'You say goodbye and I say hello.'
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    # print(words)

    # 创建字典创建对应单词ID和单词对应表
    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])
    corpus = np.array(corpus)
    # print(corpus)

    return corpus, word_to_id, id_to_word


# 基于计数的方法
# 也就是获取每个单词的共现矩阵（根据窗口大小）
def create_co_matrix(corus, vocab_size, window_size=1):
    corpus_size = len(corus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        # 向左向右提取 window_size 个单词
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            # 必须要在矩阵范围内才能加
            if left_idx >= 0:
                co_matrix[word_id, corpus[left_idx]] += 1

            if right_idx < corpus_size:
                co_matrix[word_id, corpus[right_idx]] += 1

    return co_matrix


# Positive Pointwise Mutual Information
# 也就是让概论最小为0
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    # 暴力计算 字符串 的 PMI 值
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print('%.1f%% done' % (100 * cnt / total))

    return M


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
# print(corpus)
# print(word_to_id)
# print(id_to_word)


# 计算相似度
vocab_size = len(word_to_id)
# 此时 C 就是一个共先矩阵
C = create_co_matrix(corpus, vocab_size)
print(C)
c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]
print(cos_similarity(c0, c1))

most_similar('you', word_to_id, id_to_word, C, top=5)


# 转换共现矩阵为 PPMI 矩阵
W=ppmi(C)
np.set_printoptions(precision=3)
print('covariance matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)

# 基于 SVD（Singular Value Decomposition） 的降维
U,S,V=np.linalg.svd(W)

# 使用对应的前 n 个数据就进行了降维
for word, word_id in word_to_id.items():
    plt.annotate(word,(U[word_id,0],U[word_id,1]))
plt.scatter(U[:,0],U[:,1],alpha=0.5)
plt.show()