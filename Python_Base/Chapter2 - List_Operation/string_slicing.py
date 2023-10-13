letters_amazon = '''
We spent several years building our own database engine,
Amazon Aurora, a fully-managed MySQL and PostgreSQL-compatible
service with the same or better durability and availability as
the commercial engines, but at one-tenth of the cost. We were
not surprised when this worked.
'''

# one-liner 写法
# x 是大字符串，q 是你想要查找的单词
# 返回 q 的上下文18个单词
# 注意，该写法多次调用相同的函数且没有判断边界
find = lambda s, q: s[s.find(q) - 18:s.find(q) + 18] if q in s else -1
print(find(letters_amazon, 'SQL'))


# 经典函数写法
# 注意，该写法多次调用相同的函数且没有判断边界
def findStr(str, word, gap=18):
    if word in str:
        return str[str.find(word) - gap:str.find(word) + gap]

    return -1


# 好的写法
def findStr2(str, word, gap=18):
    idx = str.find(word)
    return str[idx - gap:idx + gap] if word in str else -1


print(findStr(letters_amazon, 'SQL', 8))
print(findStr2(letters_amazon, 'SQL', 8))
