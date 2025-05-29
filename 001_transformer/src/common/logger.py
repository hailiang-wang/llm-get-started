#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===============================================================================
#
# Copyright (c) 2025 Hai Liang W.<hailiang.hl.wang@gmail.com> All Rights Reserved
#
#
# File: /Users/hain/chatopera/chatopera.xinli/alchemist/app/common/logger.py
# Author: Hai Liang Wang
# Date: 2020-03-21:14:44:42
#
# ===============================================================================

"""
   
"""
import logging
__copyright__ = "Copyright (c) 2025 Hai Liang W.<hailiang.hl.wang@gmail.com> All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2020-03-21:14:44:42"

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curdir, os.pardir))

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

import json
from datetime import datetime

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
print("LOG_LEVEL %s" % LOG_LEVEL)
logging.basicConfig(level=LOG_LEVEL)

# 日志级别
CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0


def get_humanreadable_timestamp():
    """
    获得Unix 时间戳
    :return: Float Number, use int(get_time_stamp()) to get Seconds Timestamp
    """
    return datetime.today().strftime('%Y/%m/%d %H:%M:%S')

class Logger(logging.Logger):
    """
    Logger
    """

    def __init__(self, name, level=DEBUG):
        self.name = name
        self.level = level
        logging.Logger.__init__(self, self.name, level=level)
        self.__setStreamHandler__()

    def __setStreamHandler__(self, level=None):
        """
        set stream handler
        :param level:
        :return:
        """
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(levelname)s %(asctime)s %(name)s.%(funcName)s@%(lineno)d: %(message)s')
        stream_handler.setFormatter(formatter)
        if not level:
            stream_handler.setLevel(LOG_LEVEL)
        else:
            stream_handler.setLevel(level)
        self.addHandler(stream_handler)


class FileLogger():
    """
    Logging to file
    """

    def __init__(self, logfile):
        '''
        Init logger
        '''
        self.logfile = logfile
        parent_dir = os.path.dirname(self.logfile)

        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

    def append(self, level, msg):
        ts = get_humanreadable_timestamp()
        with open(self.logfile, "a", encoding="utf-8") as fout:
            fmsg = "%s [%s] %s\n" % (ts, level, msg)
            fout.write(fmsg)
            print(fmsg)

    def debug(self, msg):
        self.append("DEBUG", msg)

    def info(self, msg):
        self.append("INFO", msg)

    def warn(self, msg):
        self.append("WARN", msg)

    def error(self, msg):
        self.append("ERROR", msg)

    def remove(self):
        os.remove(self.logfile)


def pretty(j, indent=4, sort_keys=True):
    """
    get json as pretty string
    :param j:
    :return:
    """
    return json.dumps(j, indent=indent, sort_keys=sort_keys, ensure_ascii=False, default=str)


def LN(x):
    """
    log name 获得模块名字，输出日志短名称
    """
    return x.split(".")[-1]
