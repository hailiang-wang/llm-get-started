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
__copyright__ = "Copyright (c) 2025 Hai Liang W.<hailiang.hl.wang@gmail.com> All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2020-03-21:14:44:42"

import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curdir, os.path.pardir))

# Environment Variables
ENVIRON = os.environ.copy()
# ENVIRON["JAVA_HOME"] = "/usr/lib/jvm/java-8-oracle"

import re
import unicodedata
import os
import sys
import subprocess
from six import string_types, u
import shutil
import logging
from contextlib import contextmanager
import numpy as np
import numbers
from datetime import date
from datetime import datetime
import hashlib

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

try:
    from html.entities import name2codepoint as n2cp
except ImportError:
    from htmlentitydefs import name2codepoint as n2cp
try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

try:
    from smart_open import smart_open
except ImportError:
    logging.debug("smart_open library not found; falling back to local-filesystem-only")

    def make_closing(base, **attrs):
        """
        Add support for `with Base(attrs) as fout:` to the base class if it's missing.
        The base class' `close()` method will be called on context exit, to always close the file properly.

        This is needed for gzip.GzipFile, bz2.BZ2File etc in older Pythons (<=2.6), which otherwise
        raise "AttributeError: GzipFile instance has no attribute '__exit__'".

        """
        if not hasattr(base, '__enter__'):
            attrs['__enter__'] = lambda self: self
        if not hasattr(base, '__exit__'):
            attrs['__exit__'] = lambda self, type, value, traceback: self.close()
        return type('Closing' + base.__name__, (base, object), attrs)

    def smart_open(fname, mode='rb'):
        _, ext = os.path.splitext(fname)
        if ext == '.bz2':
            from bz2 import BZ2File
            return make_closing(BZ2File)(fname, mode)
        if ext == '.gz':
            from gzip import GzipFile
            return make_closing(GzipFile)(fname, mode)
        return open(fname, mode)

PAT_ALPHABETIC = re.compile(r'(((?![\d])\w)+)', re.UNICODE)
RE_HTML_ENTITY = re.compile(r'&(#?)([xX]?)(\w{1,8});', re.UNICODE)


def get_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance.
    Method originally from maciejkula/glove-python, and written by @joshloyal.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        '%r cannot be used to seed a np.random.RandomState instance' %
        seed)


class NoCM(object):
    def acquire(self):
        pass

    def release(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


nocm = NoCM()


@contextmanager
def file_or_filename(input):
    """
    Return a file-like object ready to be read from the beginning. `input` is either
    a filename (gz/bz2 also supported) or a file-like object supporting seek.

    """
    if isinstance(input, string_types):
        # input was a filename: open as file
        yield smart_open(input)
    else:
        # input already a file-like object; just reset to the beginning
        input.seek(0)
        yield input


def deaccent(text):
    """
    Remove accentuation from the given string. Input text is either a unicode string or utf8 encoded bytestring.

    Return input string with accents removed, as unicode.

    >>> deaccent("Šéf chomutovských komunistů dostal poštou bílý prášek")
    u'Sef chomutovskych komunistu dostal postou bily prasek'

    """
    if not isinstance(text, unicode):
        # assume utf8 for byte strings, use default (strict) error handling
        text = text.decode('utf8')
    norm = unicodedata.normalize("NFD", text)
    result = u('').join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)


def copytree_hardlink(source, dest):
    """
    Recursively copy a directory ala shutils.copytree, but hardlink files
    instead of copying. Available on UNIX systems only.
    """
    copy2 = shutil.copy2
    try:
        shutil.copy2 = os.link
        shutil.copytree(source, dest)
    finally:
        shutil.copy2 = copy2


def tokenize(
        text,
        lowercase=False,
        deacc=False,
        encoding='utf8',
        errors="strict",
        to_lower=False,
        lower=False):
    """
    Iteratively yield tokens as unicode strings, removing accent marks
    and optionally lowercasing the unidoce string by assigning True
    to one of the parameters, lowercase, to_lower, or lower.

    Input text may be either unicode or utf8-encoded byte string.

    The tokens on output are maximal contiguous sequences of alphabetic
    characters (no digits!).

    >>> list(tokenize('Nic nemůže letět rychlostí vyšší, než 300 tisíc kilometrů za sekundu!', deacc = True))
    [u'Nic', u'nemuze', u'letet', u'rychlosti', u'vyssi', u'nez', u'tisic', u'kilometru', u'za', u'sekundu']

    """
    lowercase = lowercase or to_lower or lower
    text = to_unicode(text, encoding, errors=errors)
    if lowercase:
        text = text.lower()
    if deacc:
        text = deaccent(text)
    return simple_tokenize(text)


def simple_tokenize(text):
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()


def simple_preprocess(doc, deacc=False, min_len=2, max_len=15):
    """
    Convert a document into a list of tokens.

    This lowercases, tokenizes, de-accents (optional). -- the output are final
    tokens = unicode strings, that won't be processed any further.

    """
    tokens = [
        token for token in tokenize(doc, lower=True, deacc=deacc, errors='ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens


def any2utf8(text, errors='strict', encoding='utf8'):
    """Convert a string (unicode or bytestring in `encoding`), to bytestring in utf8."""
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')


to_utf8 = any2utf8


def any2unicode(text, encoding='utf8', errors='strict'):
    """Convert a string (bytestring in `encoding` or unicode), to unicode."""
    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)


to_unicode = any2unicode


def call_on_class_only(*args, **kwargs):
    """Raise exception when load methods are called on instance"""
    raise AttributeError('This method should be called on a class object.')


def is_zhs(str):
    '''
    Check if str is Chinese Word
    '''
    for i in str:
        if not is_zh(i):
            return False
    return True


def has_zh(str):
    '''
    Check if str has Chinese Word
    '''
    for i in str:
        if is_zh(i):
            return True
    return False


def is_zh(ch):
    """return True if ch is Chinese character.
    full-width puncts/latins are not counted in.
    """
    x = ord(ch)
    # CJK Radicals Supplement and Kangxi radicals
    if 0x2e80 <= x <= 0x2fef:
        return True
    # CJK Unified Ideographs Extension A
    elif 0x3400 <= x <= 0x4dbf:
        return True
    # CJK Unified Ideographs
    elif 0x4e00 <= x <= 0x9fbb:
        return True
    # CJK Compatibility Ideographs
    elif 0xf900 <= x <= 0xfad9:
        return True
    # CJK Unified Ideographs Extension B
    elif 0x20000 <= x <= 0x2a6df:
        return True
    else:
        return False


def is_punct(ch):
    x = ord(ch)
    # in no-formal literals, space is used as punctuation sometimes.
    if x < 127 and ascii.ispunct(x):
        return True
    # General Punctuation
    elif 0x2000 <= x <= 0x206f:
        return True
    # CJK Symbols and Punctuation
    elif 0x3000 <= x <= 0x303f:
        return True
    # Halfwidth and Fullwidth Forms
    elif 0xff00 <= x <= 0xffef:
        return True
    # CJK Compatibility Forms
    elif 0xfe30 <= x <= 0xfe4f:
        return True
    else:
        return False


def exec_cmd(cmd, cwd=os.getcwd()):
    '''
    exec a string as shell scripts
    return
    '''
    # console_log("exec_cmd cwd[%s]: %s" % (cwd, cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=ENVIRON, cwd=cwd)
    out, err = p.communicate()
    # if out:
    #     console_log("  exec_cmd result[out]: %s" % out)
    if err:
        console_log("  exec_cmd result[err]: %s" % err)
    return out, err


def shell_cp(source, target):
    """
    复印文件或文件夹
    """
    return exec_cmd('cp -rf %s %s' % (source, target))


def create_dir(target, remove=False):
    '''
    Create a folder.
    return: stdout, stderr
    '''
    rmstr = ""
    if os.path.exists(target):
        if remove:
            rmstr = "rm -rf %s &&" % target
        else:
            return None
    return exec_cmd('%smkdir -p %s' % (rmstr, target))


def get_timestamp():
    """
    获得Unix 时间戳
    :return: Float Number, use int(get_time_stamp()) to get Seconds Timestamp
    """
    now = datetime.now()
    return datetime.timestamp(now)


def get_humanreadable_timestamp():
    """
    获得Unix 时间戳
    :return: Float Number, use int(get_time_stamp()) to get Seconds Timestamp
    """
    return datetime.today().strftime('%Y_%m_%d_%H%M%S')


def get_humanreadable_datetime(fmt="%Y-%m-%d %H:%M:%S"):
    currentDT = datetime.now()
    return currentDT.strftime(fmt)


def parse_datetime_from_ts(timestamp):
    """
    将时间戳转化为datetime
    :param timestamp:
    :return:
    """
    return datetime.fromtimestamp(timestamp)


def add_date_years(d, years):
    """Return a date that's `years` years after the date (or datetime)
    object `d`. Return the same calendar date (month and day) in the
    destination year, if it exists, otherwise use the following day
    (thus changing February 29 to March 1).

    """
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d + (date(d.year + years, 3, 1) - date(d.year, 3, 1))


def console_log(line):
    '''
    print log for status up during indexing
    '''
    print("%s %s" % (get_humanreadable_datetime(), line))


def replace_last_occr_in_string(target, str_to_be_replaced, replacement):
    '''
    Replace Last occurrence of a String
    '''
    # Reverse the substring that need to be replaced
    str_to_be_replaced_reverse = str_to_be_replaced[::-1]
    # Reverse the replacement substring
    replacement_reverse = replacement[::-1]

    return target[::-1].replace(str_to_be_replaced_reverse, replacement_reverse, 1)[::-1]


def get_substring_as_tail(target, num):
    '''
    Get substring with tail
    e.g. get_substring_as_tail("123456", 1) --> "6"
    '''

    return target[::-1][0:num][::-1]

def hash_string(uttr):
    '''
    generate md5 based on input text
    '''
    return hashlib.md5(uttr.encode()).hexdigest()