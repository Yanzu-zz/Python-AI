# Dependencies
import numpy as np

# Data (row = [title, rating])
books = np.array([['Coffee Break NumPy', 4.6],
                  ['Lord of the Rings', 5.0],
                  ['Harry Potter', 4.3],
                  ['Winnie-the-Pooh', 3.9],
                  ['The Clown of God', 2.2],
                  ['Coffee Break Python', 4.7]])

# 与数字对比之前记得转换数据类型
findBestSeller = lambda x, y: x[x[:, 1].astype(np.float16) > y]

print(findBestSeller(books, 3.9))
print(findBestSeller(books, 3.9)[:, 0])
