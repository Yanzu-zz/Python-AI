import numpy as np

import sys, os

sys.path.append(os.pardir)
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100

# 加载 ptb 数据集
corpus, word_to_id, id_to_word = ptb.load_data('train')
# ptb 数据集的大小（10000）
vocab_size = len(word_to_id)
# print(vocab_size)

# 统计共现矩阵
print('counting co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)

# 创建 PPMI 矩阵
print('calculating PPMI ...')
W = ppmi(C, verbose=True)

# SVD 降维
print('calculating SVD ...')
try:
    # truncated SVD (fast!)
    # 使用随机数的 Truncated SVD，仅对奇异值较大的部分进行计算，计算速度比常规的 SVD 快
    # 剩余的代码和之前使用小语料库时的代码差不太多
    from sklearn.utils.extmath import randomized_svd

    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
except ImportError:
    # SVD (slow)
    # 对全数据进行 SVD 降维
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]
queries = ['you', 'year', 'car', 'toyota']
for query in queries:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
