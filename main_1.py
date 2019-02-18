# -*- coding: utf-8 -*-


from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time


# 训练模型
t0 = int(time.time())
sentences = LineSentence('wiki.zh.word.text')
# size：词向量长度
# window：词向量上下文最大距离，默认为 5，对于一般的语料，这个值推荐在 [5, 10] 之间
# sg：如果设置为 0，则是 CBOW 模型，如果是 1，则是 Skip-Gram 模型，默认为 0
# hs：如果是 0，则是负采样（Negative Sampling），如果是 1，则是 Hierarchical Softmax，默认为 0
# negative：即使用 Negative Sampling 时负采样的个数，默认是 5，推荐在 [3, 10] 之间
# min_count：需要计算词向量的最小词频，这个值可以去掉一些很生僻的低频词，默认值为 5
# iter：随机梯度下降法中迭代的最大次数，默认是 5
# alpha：在随机梯度下降法中迭代的初始步长，默认是0.025
# min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha 给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由 iter、alpha、min_alpha 一起得出
model = Word2Vec(sentences, size=128, window=5, min_count=5, workers=4)
print('训练耗时 %d s' % (int(time.time()) - t0))

# 保存模型
model.save('gensim_128')

