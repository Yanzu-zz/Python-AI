from torch.utils.data import Dataset, DataLoader
import numpy as np
import jieba


def read_dict(voc_dict_path):
    voc_dict = {}
    # voc_dict = {line.split(",")[0]: line.split(",")[1].strip() for line in open(voc_dict_path).readlines()}
    dict_list = open(voc_dict_path, encoding="utf-8").readlines()

    for item in dict_list:
        item = item.split(",")
        voc_dict[item[0]] = int(item[1].strip())

    return voc_dict


def load_data(data_path, data_stop_path):
    # 读取停用词（这些词不计入频率）
    stops_word = [line.strip() for line in open(data_stop_path, encoding="utf-8").readlines()]
    stops_word.append(" ")
    stops_word.append("\n")

    # 拿到微博评论数据
    data_list = open(data_path, encoding="utf-8").readlines()[1:]

    # 接着分词
    voc_dict = {}
    data = []
    max_len_seq = 0
    for item in data_list[:150]:
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

        max_len_seq = max(max_len_seq, len(seg_res))
        data.append([label, seg_res])
    return data, max_len_seq


class textCls(Dataset):
    def __init__(self, voc_dict_path, data_path, data_stop_path):
        self.data_path = data_path
        self.data_stop_path = data_stop_path
        self.voc_dict = read_dict(voc_dict_path)
        self.data, self.max_len_seq = load_data(data_path, data_stop_path)

        # shuffle data
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    # 获取数据
    def __getitem__(self, item):
        data = self.data[item]
        label = int(data[0])
        word_list = data[1]
        input_idx = []

        for word in word_list:
            if word in self.voc_dict.keys():
                input_idx.append(self.voc_dict[word])
            else:
                input_idx.append(self.voc_dict["<UNK>"])

        # 长度不够就 PAD
        if len(input_idx) < self.max_len_seq:
            input_idx += [self.voc_dict["<PAD>"]
                          for _ in range(self.max_len_seq - len(input_idx))]

        data = np.array(input_idx)
        return label, data


def data_loader(dataset, config):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_shuffle)


if __name__ == "__main__":
    data_path = "./sources/weibo_senti_100k.csv"
    data_stop_path = "./sources/hit_stopword"
    dict_path = "./sources/dict"
    dataloader = data_loader(data_path, data_stop_path, dict_path)

    for i, batch in enumerate(dataloader):
        print(batch)
