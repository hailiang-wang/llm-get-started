'''
https://docs.pytorch.ac.cn/docs/2.10/generated/torch.nn.Embedding.html

在之前，已经将一个句子转化为了一个 [word_ids], 比如 [0,90000,9,3,1,2,2,2,2 ...]
--> 需要将这个句子，转换为多维的张量，而且是一个紧凑的矩阵，而不是稀疏的矩阵
--> 数字，都转化为 -2~2 这个区间。
'''

import torch
import torch.nn as nn


# 2000 单词 -onehot 编码-> 每个单词的维度是多少？ [0,1, ... 1998 个 0]
# 2000 单词 -embeding 编码 -> 每个单词，维度是 10? [0.1, 0.24, 0.56, ... 还有 7 个小数]

# N 是单词的数，比如词汇表中有 2000 个单词，每个单词都是不一样的，那么 n = 2000
# d 是每个单词，用多少维的向量表达，比如每个单词，映射为 10 维度的向量
n, d, m = 2000, 10, 7
embedding = nn.Embedding(n, d, max_norm=1.0)

input = torch.LongTensor([[1, 2, 4, 1999], [4, 3, 200, 9]])
output = embedding(input)
print(output)
print(output.shape)  # torch.Size([2, 4, 10])
