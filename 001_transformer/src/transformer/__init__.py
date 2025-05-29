
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2025 <> All Rights Reserved
#
#
# File: /c/Users/Administrator/courses/llms/transformer-pytorch-get-started/src/tranformer.py
# Author: Hai Liang Wang
# Date: 2025-04-17:13:53:42
#
#===============================================================================

"""
   
"""
__copyright__ = "Copyright (c) 2020 . All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2025-04-17:13:53:42"

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(curdir, os.pardir))

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

import torch.nn as nn
from torchinfo import summary
import copy
from transformer.network import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, EncoderDecoder, Encoder, EncoderLayer, Decoder, DecoderLayer, Embeddings, Generator

log = lambda x, y: print(x) if y is None else y.info(x)

def make_model(
    src_vocab, 
    tgt_vocab, 
    N=6, 
    d_model=512, 
    d_ff=2048, 
    h=8, 
    dropout=0.1,
    logger=None
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    log(summary(model, verbose=0, depth=6), logger)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
