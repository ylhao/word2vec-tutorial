# -*- coding: utf-8 -*-


from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time


# 加载模型
model = Word2Vec.load('gensim_128')

# 找出“数学”的相关词
items = model.wv.most_similar('数学')
print('=' * 100)
print("model.wv.most_similar('数学')：")
for i, item in enumerate(items):
    # item[0]: 词
    # item[1]: 相似度
    print(i + 1, item[0], item[1])

print('=' * 100)
print("model.wv.most_similar(positive=['中国', '纽约'], negative=['北京']):")
items = model.wv.most_similar(positive=['中国', '纽约'], negative=['北京'])
for i, item in enumerate(items):
    print(i, item[0], item[1])

# 找出一组词中与其它词相关度最低的词
print('=' * 100)
print("model.wv.doesnt_match(['早餐', '午餐', '晚餐', '手机']):")
print(model.wv.doesnt_match(['早餐', '午餐', '晚餐', '手机']))

# 计算两个词的相关度
print('=' * 100)
print("model.wv.similarity('男人', '女人'):")
print(model.wv.similarity('男人', '女人'))
