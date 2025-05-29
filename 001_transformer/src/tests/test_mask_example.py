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

from transformer.helpers import subsequent_mask, show_example
import pandas as pd
import altair as alt

# Get ENV
ENVIRON = os.environ.copy()
alt.renderers.enable("browser")


def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


example_mask().show()