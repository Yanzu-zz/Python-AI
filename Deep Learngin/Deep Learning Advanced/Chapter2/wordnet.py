# 第一次使用需要下载 wordnet 资源
# import nltk
# nltk.download('wordnet')

from nltk.corpus import wordnet

# 获得 car 的同义词
print(wordnet.synsets('car'))

# 同义词簇
car = wordnet.synset('car.n.01')
print(car.definition())
# 获取 car 的同义词
print(car.lemma_names())

print(car.hypernym_paths()[0])

# 基于WordNet的语义相似度
novel = wordnet.synset('novel.n.01')
dog = wordnet.synset('dog.n.01')
motorcycle = wordnet.synset('motorcycle.n.01')
# 比较 car 和 novel 的相似度
print(car.path_similarity(novel))
print(car.path_similarity(dog))
print(car.path_similarity(motorcycle))
