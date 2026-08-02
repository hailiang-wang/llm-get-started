'''
https://medium.com/@smrati.katiyar/explaining-embedding-layer-in-pytorch-1f22b88c1a69

https://docs.pytorch.ac.cn/docs/2.11/generated/torch.nn.Embedding.html
'''
import torch
import torch.nn as nn

# Define an Embedding layer
# OneHot 编码
# 你  - 1 - [1, 0, 0, 97 个0]
# 我  -  2 -[0, 1, 0, 97 个0]
# ...
# 他 - 100 - [前面是 99 个  0 ，1]

# 词嵌入编码 - Word2vec
# 你 - 1 - [0.1, 0.2, 0.5]
# 我 - 2 - [0.1, 0.3, 0.5]
# 菠萝 - 99 - [0.5, 0.1, 1.2]
embedding_layer = nn.Embedding(num_embeddings=11, embedding_dim=3)

# Example input (batch of token indices)
input_indices = torch.tensor([1, 2, 3, 10, 8])
# ---> [1, 2, 3, 4] 是一句话，是由 [wordId1, wordId2, wordId3, wordId4] 转换来的

# Forward pass through the embedding layer
output = embedding_layer(input_indices)

print(output)
