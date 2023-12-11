#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from hashlib import sha1
from typing import Any
import re


def slugify(value: Any) -> str:
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    - Convert spaces or repeated dashes to single dashes.
    - Remove characters that aren't alphanumerics,
        underscores, or hyphens.
    - Convert to lowercase.
    - Also strip leading and trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def sha1_hash(data: bytes, d: int = 32) -> int:
    """
    Generate a d-bit hash value from the given data.
    Parameters
    ----------
    data : bytes
        The data to be hashed.
    d : int
        The number of bits of the hash value.
    Returns
    -------
    int
        The hash value.
    Examples
    --------
    >>> sha1_hash(b"hello world", 32)
    896314922
    >>> sha1_hash(b"hello world", 64)
    13028719972609469994
    """
    return int.from_bytes(sha1(data).digest()[: d // 8], byteorder="little")
