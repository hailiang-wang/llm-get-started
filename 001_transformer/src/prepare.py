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

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

import shutil
import json

# Get ENV
from env import ENV
from common.utils import get_humanreadable_timestamp

src_parent_dir = os.path.join(curdir, os.pardir)

HYPER_PARAMS_JSON="hyper_params.json"
DATASET_ROOT = ENV.str("DATASET_ROOT", os.path.join(src_parent_dir, "data"))
RESULT_ROOT = ENV.str("RESULT_ROOT", os.path.join(src_parent_dir, "result"))
TMPDIR_ROOT = ENV.str("TMPDIR_ROOT", os.path.join(src_parent_dir, "tmp"))

print(">> DATASET_ROOT", DATASET_ROOT)
print(">> RESULT_ROOT", RESULT_ROOT)
print(">> TMPDIR_ROOT", TMPDIR_ROOT)

if not os.path.exists(DATASET_ROOT):
    os.makedirs(DATASET_ROOT)
if not os.path.exists(RESULT_ROOT):
    os.makedirs(RESULT_ROOT)
if not os.path.exists(TMPDIR_ROOT):
    os.makedirs(TMPDIR_ROOT)

def generate_new_resultdir(framework="default"):
    '''
    Return a new result dir
    params:
        * framework: method, algm, e.g. BERT, RNN, transformer.
    '''
    newdir = os.path.join(RESULT_ROOT, framework, get_humanreadable_timestamp())

    if not os.path.exists(newdir):
        os.makedirs(newdir)

    return newdir

def copy_env(envfile, resultdir):
    '''
    Copy .env file
    '''
    shutil.copy(envfile, resultdir)

def dump_hyper_params(hyper_params, resultdir):
    '''
    Dump hyper params
    '''
    with open(os.path.join(resultdir, HYPER_PARAMS_JSON), "w") as fout:
        json.dump(hyper_params, fout, ensure_ascii=False, indent=4)


def read_hyper_params(resultdir):
    '''
    Read 
    '''
    result = None
    with open(os.path.join(resultdir, HYPER_PARAMS_JSON), "r") as fin:
        result = json.load(fin)
    
    return result


##########################################################################
# Testcases
##########################################################################
import unittest

# run testcase: python /c/Users/Administrator/courses/llms/transformer-pytorch-get-started/src/prepare.py Test.testExample
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
    #suite = unittest.TestSuite()
    #suite.addTest(Test("test_001"))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)

    # run as main, accept pass testcase name with argvs
    unittest.main()

def main():
    test()

if __name__ == '__main__':
    main()
