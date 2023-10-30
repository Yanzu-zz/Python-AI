import jieba
from utils import normalizeString, cht_to_chs

SOS_token = 0
EOS_token = 0
# 不要太长的句子（我们这就一练习项目）
MAX_LENGTH = 10


# 对不同语言的文本进行统计
# 完成词转索引，字典统计等功能
class Lang():
    def __init__(self, name):
        # name 判断是中文还是英文
        self.name = name

        # 词的索引
        self.word2index = {}
        # 词的数量（也就是词频统计）
        self.word2count = {}
        # 反转
        self.index2word = {
            # 起止符和终止符
            0: "SOS",
            1: "EOS"
        }
        # 统计有几个不同的单词
        self.n_words = 2

    # 添加一个词
    def addWord(self, word):
        if word not in self.word2index:
            # 双向存储
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        # 添加句子中每个词
        for word in sentence.split(" "):
            self.addWord(word)


# 文本解析
def readLangs(lang1, lang2, path):
    lines = open(path, encoding="utf-8").readlines()
    pairs = []

    lang1_cls = Lang(lang1)
    lang2_cls = Lang(lang2)

    for l in lines:
        # 第三列是贡献者信息，忽略掉
        l = l.split("\t")
        # 格式化一下英文串
        sentence1 = normalizeString(l[0])
        # 中文翻译有些是繁体字，转换一下它
        sentence2 = cht_to_chs(l[1])

        # 分词
        seg_list = jieba.cut(sentence2, cut_all=False)
        sentence2 = " ".join(seg_list)

        # 不要超长的句子
        if len(sentence1.split(" ")) > MAX_LENGTH:
            continue
        if len(sentence2.split(" ")) > MAX_LENGTH:
            continue

        pairs.append([sentence1, sentence2])
        lang1_cls.addSentence(sentence1)
        lang2_cls.addSentence(sentence2)

    return lang1_cls, lang2_cls, pairs


if __name__ == "__main__":
    lang1 = "en"
    lang2 = "cn"
    path = "./data/en-cn.txt"
    lang1_cls, lang2_cls, pairs = readLangs(lang1, lang2, path)

    print(len(pairs))
    print(lang1_cls.n_words)
    print(lang1_cls.index2word)

    print(lang2_cls.n_words)
    print(lang2_cls.index2word)
