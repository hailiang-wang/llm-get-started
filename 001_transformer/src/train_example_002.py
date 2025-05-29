#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Modification Right (c) 2025 Hai Liang W.<hailiang.hl.wang@gmail.com> . Licensed under the Apache License, Version 2.0
# Copyright (c) 2018 Alexander Rush, MIT License, published with https://nlp.seas.harvard.edu/annotated-transformer/
#
# File: /c/Users/Administrator/courses/llms/transformer-pytorch-get-started/src/test_mask_example.py
# Author: Hai Liang Wang
# Date: 2025-04-17:15:14:05
#
#===============================================================================

"""
   
"""
__copyright__ = "Copyright (c) 2025 Hai Liang W.<hailiang.hl.wang@gmail.com> . Licensed under the Apache License, Version 2.0"
__author__ = "Hai Liang Wang"
__date__ = "2025-04-17:15:14:05"

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

from os.path import exists
import torch
from torch.optim.lr_scheduler import LambdaLR
import spacy
import touchtext.datasets as datasets
from touchtext.vocab import build_vocab_from_iterator
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.functional import pad
from touchtext.data.functional import to_map_style_dataset
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
import GPUtil
import torch.multiprocessing as mp
import altair as alt

# Get ENV
from env import ENV, ENV_LOCAL_RC
from transformer import make_model
from transformer.train import Batch, LabelSmoothing, rate, run_epoch, TrainState, SimpleLossCompute, greedy_decode
from transformer.helpers import DummyOptimizer, DummyScheduler
import transformer.visual as visual
from common.logger import FileLogger
from prepare import generate_new_resultdir, copy_env, dump_hyper_params

RESULT_DIR = generate_new_resultdir(framework="transformer")
VOCAB_PT_FILEPATH = os.path.join(RESULT_DIR, "vocab.pt")
MODEL_PT_FILEPATH = os.path.join(RESULT_DIR, "multi30k_model_final.pt")

logger = FileLogger(os.path.join(RESULT_DIR, "train.log"))
logger.info("RESULT_DIR %s" % RESULT_DIR)

# To display altair with browser
# TODO, further save these images to disk RESULT dir
alt.renderers.enable("browser")

# Load spacy tokenizer models, download them if they haven't been
# downloaded already
def load_tokenizers():

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)

def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    # downloaded to C:\Users\Administrator\.cache\torch\text\datasets\Multi30k automactically.
    train_iter, valid_iter, test_iter = datasets.Multi30k(
        language_pair=("de", "en")
    )

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    logger.info("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    logger.info("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists(VOCAB_PT_FILEPATH):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), VOCAB_PT_FILEPATH)
    else:
        vocab_src, vocab_tgt = torch.load(VOCAB_PT_FILEPATH)
    logger.info("Finished.\nVocabulary sizes:")
    logger.info(len(vocab_src))
    logger.info(len(vocab_tgt))
    return vocab_src, vocab_tgt

def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    hyper_params,
    is_distributed=False,
):
    logger.info(f"Train worker process using GPU: {gpu} for training")
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6,logger=logger)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=hyper_params["batch_size"] // ngpus_per_node,
        max_padding=hyper_params["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyper_params["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=hyper_params["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(hyper_params["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        logger.info(f"[GPU{gpu}] Epoch {epoch} Training ====")
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=hyper_params["accum_iter"],
            train_state=train_state,
            logger=logger
        )

        GPUtil.showUtilization()
        # skip temp model files to save space and boost time
        # if is_main_process:
        #     file_path = os.path.join(RESULT_DIR, "%s%.2d.pt" % (hyper_params["file_prefix"], epoch))
        #     torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        logger.info(f"[GPU{gpu}] Epoch {epoch} Validation ====")
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        logger.info(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        torch.save(module.state_dict(), MODEL_PT_FILEPATH)


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, hyper_params):

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    logger.info(f"Number of GPUs detected: {ngpus}")
    logger.info("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, hyper_params, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, hyper_params):
    if hyper_params["distributed"]:
        train_distributed_model(
            vocab_src, vocab_tgt, spacy_de, spacy_en, hyper_params
        )
    else:
        train_worker(
            0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, hyper_params, False
        )


def load_trained_model(vocab_src, vocab_tgt, spacy_de, spacy_en):
    hyper_params = {
        "batch_size": ENV.int("HYPER_PARAM_BATCHSIZE", 32),
        "distributed": ENV.bool("HYPER_PARAM_DISTRIBUTED", False),
        "num_epochs": ENV.int("HYPER_PARAM_EPOCH", 10),
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    dump_hyper_params(hyper_params,resultdir=RESULT_DIR)

    if not exists(MODEL_PT_FILEPATH):
        copy_env(ENV_LOCAL_RC, RESULT_DIR)
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, hyper_params)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load(MODEL_PT_FILEPATH))

    return model


def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        logger.info("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [
            vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx
        ]
        tgt_tokens = [
            vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx
        ]

        logger.info(
            "Source Text (Input)        : "
            + " ".join(src_tokens).replace("\n", "")
        )
        logger.info(
            "Target Text (Ground Truth) : "
            + " ".join(tgt_tokens).replace("\n", "")
        )
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        logger.info("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results

def run_model_example(vocab_src, vocab_tgt, spacy_de, spacy_en, device="cpu", n_examples=5):
    logger.info("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    logger.info("Loading Trained Model ...")
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load(MODEL_PT_FILEPATH, map_location=torch.device(device))
    )

    logger.info("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data

def main():
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    model = load_trained_model(vocab_src, vocab_tgt, spacy_de, spacy_en)

    '''
    # Additional Components: BPE, Search, Averaging
    2. Shared Embeddings: When using BPE with shared vocabulary we can share the same weight vectors between the source / target / generator. See the (cite) for details. To add this to the model simply do this:
    '''
    if False:
        model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
        model.generator.lut.weight = model.tgt_embed[0].lut.weight

    '''

    > 4) Model Averaging: The paper averages the last k checkpoints to
    > create an ensembling effect. We can do this after the fact if we
    > have a bunch of models:
    '''
    def average(model, models):
        "Average models into model"
        for ps in zip(*[m.params() for m in [model] + models]):
            ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


    logger.info("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cuda"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    model, example_data = run_model_example(vocab_src, vocab_tgt, spacy_de, spacy_en, device="cpu", n_examples=5)
    visual.viz_encoder_self(model, example_data).show()
    visual.viz_decoder_self(model, example_data).show()
    visual.viz_decoder_src(model, example_data).show()


if __name__ == '__main__':
    main()

