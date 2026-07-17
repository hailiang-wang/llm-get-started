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

import time
import torch
import torch.nn as nn

from transformer.helpers import subsequent_mask


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # 1x1x72

        print("src_mask shape", self.src_mask.shape)
        if tgt is not None:
            print("tgt shape", tgt.shape)  # 32x72

        if tgt is not None:
            # -1 的位置，大部是 <blank> pad; self.tgt 里，是有 <\s>
            self.tgt = tgt[:, :-1]  # 句子的第一个token 是 <s> 开始
            self.tgt_y = tgt[:, 1:]
            # print("self.tgt[0]", self.tgt[0].tolist())
            # print("self.tgt_y[0]", self.tgt_y[0].tolist())

            # print("self.tgt_y", self.tgt_y.shape)

            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
            # print("self.ntokens", self.ntokens.item())

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)

        # print("tgt_mask shape", tgt_mask.shape)
        # # print("tgt_mask shape", tgt_mask.tolist())

        subsequent_mask_t = subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        print("*" * 80)
        print("tgt_mask.shape before subsequent_mask", tgt_mask.shape)
        print("tgt_mask before subsequent_mask", tgt_mask)

        print("*" * 80)

        print("subsequent_mask", subsequent_mask_t)
        tgt_mask = tgt_mask & subsequent_mask_t
        # print("tgt_mask.data", tgt_mask.data)
        print("*" * 80)
        print("tgt_mask.shape after subsequent_mask", tgt_mask.shape)
        return tgt_mask


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def log(x, y): return print(x) if y is None else y.info(x)


def run_epoch(
    data_iter,
    model,
    loss_compute,  # SimpleLossCompute(module.generator, criterion),
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
    logger=None
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        # batch.src(1x72), batch.tgt(1x71), batch.src_mask(1x1x72), batch.tgt_mask(1x71x71)
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        # out shape [1x71x512], tgt_y 是 src 的对应的目标翻译句子， tgt_y shape [1, 71]
        # print("batch.ntokens ", batch.ntokens)
        # batch.ntokens  tensor(11, device='cuda:0')
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            log(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr), logger
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        '''
        size: len(vocab_tgt), 目标翻译语言的词汇表的大小；6384 for english Multi30k
        padding_idx: blank word id, 2
        '''

        super(LabelSmoothing, self).__init__()
        # https://docs.pytorch.org/docs/2.12/generated/torch.nn.KLDivLoss.html
        # TODO 问题：
        #    1. 为什么不使用 MSE 或 NLL Loss 呢？
        #    2. KLDivLoss 好处，应用场景
        # 深度学习常见几种损失函数以及适用场景
        # 一个分享：https://zhuanlan.zhihu.com/p/666303929
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        '''
        x shape 71x6384
        target shape 71
        为什么是 71? 因为 batch 只有 1 个，而每个句子是 71 个单词（原来是 72，去掉了 第一个 <s> 不需要预测）
        如果 batch 是 2 的话，那么
         - x shape 142x6384
         - target shape 142
        '''
        assert x.size(1) == self.size
        # 不同拷贝方法介绍：https://www.geeksforgeeks.org/deep-learning/way-to-copy-a-tensor-in-pytorch/
        true_dist = x.data.clone()

        # #REF204 此处，将 wordid 转化为了 onehot 向量，至关重要
        # 因为在推理阶段，将 softmax 的最大概率的索引，看成是 wordid，然后查找对应的
        # 词语，其关键就是这里，将词语转化为了 onehot 向量，实现了 wordid 到 onehot
        # 分布的一一对应
        # 将 true_dist 的所有数字都设置为 smoothing 值，默认 smoothing = 0，所以默认的 true_dist 都变成了 0
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        # true_dist shape 71x6384, true_dist 是 target 的 one hot 模式的张量
        # 对输入 tgt 上的 blank (pad 生成的)进行掩码，忽略该部分对 loss 的影响
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, ntokens):
        '''
        ntokens: normalize values
        '''
        x = self.generator(x)
        # x shape torch.Size([1, 71, 6384])
        x = x.contiguous().view(-1, x.size(-1))
        y = y.contiguous().view(-1)
        # print("x shape", x.shape)
        # print("y shape", y.shape)
        # x shape torch.Size([71, 6384])
        # y shape torch.Size([71])

        sloss = (
            self.criterion(
                x, y
            )
            / ntokens
        )
        return sloss.data * ntokens, sloss


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])

        # 问题：为什么用 softmax 生成的索引，可以作为输出的词的 wordid
        # 观点：见 #REF204 看损失函数的生成，是否将 wordid 和 softmax 的最大概率进行了关联
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys
