#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===============================================================================
#
# Copyright (c) 2025 Hai Liang W. <hailiang.hl.wang@gmail.com>, Apache 2.0 Licensed.
#
#
# File: /c/Users/Administrator/git/Sports-Image-Classification-YOLO-ResNet/src/env_reader.py
# Author: Hai Liang Wang
# Date: 2024-12-18:13:58:20
#
# ===============================================================================

"""
   
"""
__copyright__ = "Copyright (c) 2025 Hai Liang W. <hailiang.hl.wang@gmail.com>, Apache 2.0 Licensed."
__author__ = "Hai Liang Wang"
__date__ = "2024-12-18:13:58:20"

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

from common.environs import Env

ENV_LOCAL_RC = os.environ.get("ENV_FILE", os.path.join(curdir, os.pardir, ".env"))
if not os.path.exists(ENV_LOCAL_RC):
    raise RuntimeError("Error, config file [%s] not found." % ENV_LOCAL_RC)
else:
    print("Load environment with [%s] ..." % ENV_LOCAL_RC)

# 环境变量
# https://github.com/hailiang-wang/python-environ
ENV = Env()
ENV.read_env(ENV_LOCAL_RC, recurse=False)  # read .env file, if it exists
log_level = ENV.str("LOG_LEVEL", "INFO")
os.environ["LOG_LEVEL"] = log_level
