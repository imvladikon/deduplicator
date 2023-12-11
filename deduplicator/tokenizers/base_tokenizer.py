#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence


class BaseTokenizer:

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self._tokenize(*args, **kwargs)

    def _tokenize(self, text: str) -> Sequence[str]:
        raise NotImplementedError("abstract method")
