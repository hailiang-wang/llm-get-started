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
sys.path.insert(0, os.path.join(curdir, os.pardir))

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

import pandas as pd
import altair as alt
import torch

from transformer.train import LabelSmoothing

# Get ENV
ENVIRON = os.environ.copy()
alt.renderers.enable("browser")

# Example of label smoothing.


def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .interactive()
    )


example_label_smoothing().show()