# -*- coding: UTF-8 -*-

import jieba

data_path = "./sources/weibo_senti_100k.csv"
data_stop_path = "./sources/hit_stopword"

# 读取停用词（这些词不计入频率）
stops_word = [line.strip() for line in open(data_stop_path, encoding="utf-8").readlines()]
stops_word.append(" ")
stops_word.append("\n")

# 拿到微博评论数据
data_list = open(data_path, encoding="utf-8").readlines()[1:]

# 接着分词
voc_dict = {}
voc_list = []
min_seq = 1
top_n = 1000
UNK = "<UNK>"
PAD = "<PAD>"
for item in data_list:
    label = item[0]
    # item[1] 是逗号
    content = item[2:].strip()
    seg_list = jieba.cut(content, cut_all=False)
    seg_res = []

    for seg_item in seg_list:
        # print(seg_item)
        # 如果是停用词则跳过
        if seg_item in stops_word:
            continue

        seg_res.append(seg_item)
        # 统计词频
        if seg_item in voc_dict.keys():
            voc_dict[seg_item] += 1
        else:
            voc_dict[seg_item] = 1

    # print(content)
    # print(seg_res)

# 取词频前1000个
voc_list = sorted([_ for _ in voc_dict.items() if _[1] > min_seq],
                  key=lambda x: x[1],
                  reverse=True)[:top_n]
# 重新排列
voc_dict = {word_count[0]: idx for idx, word_count in enumerate(voc_list)}
voc_dict.update({UNK: len(voc_dict), PAD: len(voc_dict) + 1})
# print(voc_list)

ff = open("./sources/dict", "w")
# 写入 键值：出现频率 格式
for item in voc_dict.keys():
    ff.writelines("{},{}\n".format(item, voc_dict[item]))
