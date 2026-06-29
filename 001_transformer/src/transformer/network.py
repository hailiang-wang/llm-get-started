#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===============================================================================
#
# Modification Right (c) 2025 Hai Liang W.<hailiang.hl.wang@gmail.com> . Licensed under the Apache License, Version 2.0
# Copyright (c) 2018 Alexander Rush, MIT License, published with https://nlp.seas.harvard.edu/annotated-transformer/
#
# File: /c/Users/Administrator/courses/llms/transformer-pytorch-get-started/src/test_mask_example.py
# Author: Hai Liang Wang
# Date: 2025-04-17:15:14:05
#
# ===============================================================================

"""
   
"""
__copyright__ = "Copyright (c) 2025 Hai Liang W.<hailiang.hl.wang@gmail.com> . Licensed under the Apache License, Version 2.0"
__author__ = "Hai Liang Wang"
__date__ = "2025-04-17:15:14:05"

import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(curdir, os.pardir))

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import math


from transformer.helpers import clones, LayerNorm

# Get ENV
from env import ENV


# Set to False to skip notebook execution (e.g. for debugging)
RUN_EXAMPLES = True


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        print("[network] shape of src %s, tgt %s, src_mask %s, tgt_mask %s" %
              (src.shape, tgt.shape, src_mask.shape, tgt_mask.shape))

        print("tgt_mask", tgt_mask)
        # Batch size = 1
        # [network] shape of src 1x72, tgt 1x71,
        # src_mask 1x1x72, tgt_mask 1x71x71
        #
        ###############
        # 背景信息
        ###############
        # 有一个翻译的任务，是从德语翻译成英语，比如有两个语句：
        # src: Drei Kinder stehen vor zwei großen Reifen.
        # tgt: Three children stand in front of two large tires.
        # 在计算到这行的时候，src 和 tgt 都是对应的 word ids, 比如
        # src: [0, 60, 67, 54, 28, 73, 80, 912, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        # 0 代表 <s>，是句子的开始
        # 1 代表 </s>，是句子的结束
        # 2 代表 <blank> 是空格，此处也用于 pad 补齐, 这个模型限制了最长输入是 72 个词语，不够的补齐，长的截断
        # 此外，还有一个特殊的符号 <unk> id 是 4，代表未知的词语，根据词频统计后，一些出现了1次的词语，被设置为 unk，或者将来在实际应用当中，没有出现在 vocab 当中的词语
        #
        ###############
        # 训练的过程
        ###############
        # 第一步，进行 encoder 的 forward 计算，在 encoder 的多头注意力中，将一句话，表示为了 1x72x512，然后
        #        被拆解为了 8 个头（思考：为什么是 8 个头？，比如知乎一下：transformer encoder 拆解为 8 个头的原因）
        #        参考阅读，8 个 head 的原因：https://zhuanlan.zhihu.com/p/2031356292967687042
        #        猜想：将一句话，认为有 8 个最主要的词语，让词语和句子本身做点乘，找到共振的方向，实现词语和词语之间的相似度计算
        #        通过这样的一个计算，来表征这个句子，得到语义空间的向量，并且，相似的句子和句子，可以有更近的距离
        # 第二步，将 encoder 提取出来的输出，作为语义特征矩阵 memory，然后，在 decoder 中，将 memory 作为 Key
        #        和 value 进行 attention 计算。
        # 第三步，decoder 中的 attention 计算，是将 tgt 进行了 71 个输入的 Output Embedding
        #        所以，tgt_mask 是 71x71, 也就是 tgt 的预测，是进行了 71 次，每次一个 mask
        #        每个 mask 是释放了一个词，所以，output embedding 在 Masked Multi Head attention
        #        是进行了 71 次运算，每次多了一个词。那么，就消除了状态的依赖。
        # 返回值是 1x71x512 的张量
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # x shape 1x71x512
        x = self.proj(x)
        print("[Generator] shape of x", x.shape)
        # [Generator] shape of x torch.Size([1, 71, 6384])
        return log_softmax(x, dim=-1)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."

        print("Encoder mask", mask[0].tolist())
        print("Encoder mask shape", mask.shape)

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # x 1x72x512, mask 1x72x1
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        normalized_x = self.norm(x)
        print("[decoder] return normalized_x shape", normalized_x.shape)
        # [decoder] return normalized_x shape torch.Size([1, 71, 512])
        return normalized_x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        print("Decoder tgt shape", x.shape)
        print("Decoder tgt_mask shape", tgt_mask.shape)
        print("Decoder memory shape", memory.shape)
        print("Decoder src_mask shape", src_mask.shape)

        # Decoder tgt shape torch.Size([1, 71, 512])
        # Decoder tgt_mask shape torch.Size([1, 71, 71])
        # Decoder memory shape torch.Size([1, 72, 512])
        # Decoder src_mask shape torch.Size([1, 1, 72])

        # sublayer[0] 就是 self attn, 在 self attn 中，如何利用 tgt_mask 信息的？
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y, tgt_mask))

        # self.sublayer[1] 就是 src attn
        # TODO 为什么在 Decoder 的第二个 Atten 运算中，也就是 src_atten 中，使用了 src_mask?
        print("Decoder before src_attention: decoder self attention shape %s\n  encoder attention output(memory) shape %s\n  encoder src mask shape %s" % (
            x.shape, m.shape, src_mask.shape
        ))
        # Decoder before src_attention: decoder self attention shape torch.Size([1, 71, 512])
        #   encoder attention output(memory) shape torch.Size([1, 72, 512])
        #   encoder src mask shape torch.Size([1, 1, 72])

        # TODO Decoder 中，掩码的设计和这里的 decoder self attention 的关联关系：decoder 掩码的设计，让 decoder self attention 具备了什么信息？
        # 在生成 decoder self attention 的时候，是使用了对角线矩阵，来进行掩码，而不是使用 Output 输入的句子，制作的掩码
        x = self.sublayer[1](x, lambda y: self.src_attn(y, m, m, src_mask))
        print("Decoder after src_attention: decoder attention output shape", x.shape)
        # Decoder after src_attention: decoder attention output shape torch.Size([1, 71, 512])

        # self.sublayer[2] 就是 Feedforward
        return self.sublayer[2](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    is_decoder_self_attn = False
    if (mask is not None) and (len(mask.shape) == 4) and (mask.shape[3] == 71):
        is_decoder_self_attn = True

    d_k = query.size(-1)  # dk = 64, # decoder 中 query shape 1x8x71x64
    # decoder key shape # 1x8x71x64
    # 71x64 x 64x71 = 71x71
    # 对于 Encoder 和 Decoder 而言，下面这个公式的运算，形状是类似的
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if is_decoder_self_attn:
        print("Decoder self_attn mask shape", mask.shape)
        # Decoder self_attn mask shape torch.Size([1, 1, 71, 71])
    else:
        print("Encoder self_attn mask shape", mask.shape)
        # Encoder self_attn mask shape torch.Size([1, 1, 1, 72])

    if mask is not None:
        # 这里是主要的区别：在 decoder 的 mask 中，对于这个句子中，词和词之间的投票，decoder 进行了不同时刻的遮盖
        scores = scores.masked_fill(mask == 0, -1e9)
        # print("scores shape", scores.shape) # 8x72x72
        # print("scores", scores[0][0].tolist())
    p_attn = scores.softmax(dim=-1)

    '''
    Decoder 和 Encoder 中的 self attention 得到的
    scores 肯定是有区别的，因为是两个不同的 mask 掩码结构
    那么，每个 scores 矩阵的表征是什么？
    '''
    if is_decoder_self_attn:
        print("Decoder self_attn scores shape", scores.shape)
        # Decoder self_attn scores shape torch.Size([1, 8, 71, 71]) # 71x71 的矩阵，表征到底是什么含义？是预测，还是相似度计算
        # 做了 71 词预测？
        # sys.exit(1)
    else:
        print("Encoder self_attn scores shape", scores.shape)
        # Encoder self_attn scores shape torch.Size([1, 8, 72, 72]) # 72x72 的矩阵，表征的是什么含义？是相似计算？
        # 每个单词的语义的向量？

    '''
    看起来，在 decoder 和 encoder 的 self atten 计算的最后，都是 [1,8, 句子的最大长度， 句子的最大长度]
    那么，decoder 中的 mask 影响是什么？因为，最终的形状是类似的。
    
    在自注意力下，不管是 encoder 中，还是 decoder，都是一句话的单词和单词之间进行相似度计算（因为使用的是矩阵乘法）
    那么，encoder 中，mask 是针对自然句子中，不足 72 个单词，进行掩码
    但是，decoder 中，mask 是按照顺序，从理想输出 y 中，逐个单词的进行掩码 -> 不让预测的时刻，看到后面的信息
    '''
    if dropout is not None:
        p_attn = dropout(p_attn)

    # p_attn shape # 1x8x72x72, value 1x(72)x8x64
    # print("p_attn shape", p_attn.shape) # [1, 8, 72, 72] # 72个词，和72个词之间相互投票
    # print("value shape", value.shape) # [1, 8, 72, 64]

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"

        is_decoder_self_attn = False
        is_decoder_src_attn = False

        if (mask is not None) and (len(mask.shape) == 3) and (mask.shape[1] == 71):
            is_decoder_self_attn = True

        if (mask is not None) and (query.shape[1] == 71) and (mask.shape[2] == 72):
            is_decoder_src_attn = True

        if is_decoder_src_attn:
            print("[decoder src attention] Now in decoder src attention")

        # mask shape 1x71x71
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # for decoder self attention, mask is 1x1x71x71
        nbatches = query.size(0)

        if is_decoder_self_attn is True:
            print("[decoder_self_attn] query shape", query.shape)
            print("[decoder_self_attn] query", query)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # x shape 1x72x512, lin shape 512x512
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # Decoder self attention
        if is_decoder_self_attn is True:
            print("[decoder_self_attn] mask shape", mask.shape)
            print("[decoder_self_attn] after change qkv -> query shape", query.shape)
            print("[decoder_self_attn] after change qkv -> key shape", key.shape)
            print("[decoder_self_attn] after change qkv -> value shape", value.shape)
            # [decoder_self_attn] mask shape torch.Size([1, 1, 71, 71])
            # [decoder_self_attn] after change qkv -> query shape torch.Size([1, 8, 71, 64])
            # [decoder_self_attn] after change qkv -> key shape torch.Size([1, 8, 71, 64])
            # [decoder_self_attn] after change qkv -> value shape torch.Size([1, 8, 71, 64])
        elif is_decoder_src_attn is True:
            print("[decoder_src_attn] mask shape", mask.shape)
            print("[decoder_src_attn] after change qkv -> query shape", query.shape)
            print("[decoder_src_attn] after change qkv -> key shape", key.shape)
            print("[decoder_src_attn] after change qkv -> value shape", value.shape)
            # [decoder_src_attn] mask shape torch.Size([1, 1, 1, 72])
            # [decoder_src_attn] after change qkv -> query shape torch.Size([1, 8, 71, 64])
            # [decoder_src_attn] after change qkv -> key shape torch.Size([1, 8, 72, 64])
            # [decoder_src_attn] after change qkv -> value shape torch.Size([1, 8, 72, 64])
        else:
            # Encoder self attention
            print("[encoder_self_attn] mask shape", mask.shape)
            print("[encoder_self_attn] after change qkv -> query shape", query.shape)
            print("[encoder_self_attn] after change qkv -> key shape", key.shape)
            print("[encoder_self_attn] after change qkv -> value shape", value.shape)
            # Encoder self attention
            # [encoder_self_attn] mask shape torch.Size([1, 1, 1, 72])
            # [encoder_self_attn] after change qkv -> query shape torch.Size([1, 8, 72, 64])
            # [encoder_self_attn] after change qkv -> key shape torch.Size([1, 8, 72, 64])
            # [encoder_self_attn] after change qkv -> value shape torch.Size([1, 8, 72, 64])

        # 2) Apply attention on all the projected vectors in batch.
        # attention fn 中，如何使用不同的 mask 进行计算？
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()

        # vocab --> 3W 单词的，希望可以：将一句话，输出成 512 维向量
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # print("Embedding shape", x.shape)
        # print("Embedding 0,1", x[0].tolist(), x[1].tolist())

        # batchSize(1)(pair) x 72(words) x 512(vector)
        ret = self.lut(x) * math.sqrt(self.d_model)
        # print("embeding ret shape", ret.shape)
        return ret


class PositionalEncoding(nn.Module):
    '''
    Implement the PE function.
    '''

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        print("PositionalEncoding x shape", x.shape)
        pos_info = self.pe[:, : x.size(1)].requires_grad_(False)
        print("PositionalEncoding pos_info shape", pos_info.shape)
        # 1x72x512
        x = x + pos_info
        return self.dropout(x)


##########################################################################
# Testcases
##########################################################################
import unittest

# run testcase: python /c/Users/Administrator/courses/llms/transformer-pytorch-get-started/src/tranformer.py Test.testExample


class Test(unittest.TestCase):
    '''

    '''

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_001(self):
        print("test_001")


def test():
    '''
    Run tests, two ways available
    '''

    # run as a suite
    # suite = unittest.TestSuite()
    # suite.addTest(Test("test_001"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

    # run as main, accept pass testcase name with argvs
    unittest.main()


def main():
    test()


if __name__ == '__main__':
    main()
