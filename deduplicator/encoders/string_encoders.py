#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Optional, Callable

from deduplicator.encoders.base_encoder import BaseEncoder


# similar rules to
# https://github.com/ing-bank/spark-matcher/blob/main/spark_matcher/blocker/blocking_rules.py#L159


class StringEncoder(BaseEncoder):

    def __init__(self, tokenizer: Optional[Callable] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer or self._tokenize

    def _tokenize(self, value: str) -> List[str]:
        return value.split()

    def _encode(self, value: str) -> str:
        return value


class FirstNChars(StringEncoder):

    def __init__(self, n_chars: int = 3, *args, **kwargs):
        """
        First N characters of the string
        Args:
            n_chars:
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.n_chars = n_chars

    def _encode(self, value: str) -> str:
        values = self.tokenizer(value.lower())
        values = sorted(v[: self.n_chars] for v in values)
        return "".join(values)


class FirstNCharsLastWord(StringEncoder):

    def __init__(self, n_chars: int = 3, *args, **kwargs):
        """
        First N characters of the last word of the string
        Args:
            n_chars:
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.n_chars = n_chars

    def _encode(self, value: str) -> str:
        values = self.tokenizer(value.lower())
        values = sorted(v[: self.n_chars] for v in values)
        return "".join(values[-1])


class LastNChars(StringEncoder):

    def __init__(self, n_chars: int = 3, *args, **kwargs):
        """
        Last N characters of the string
        Args:
            n_chars:
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.n_chars = n_chars

    def _encode(self, value: str) -> str:
        values = self.tokenizer(value.lower())
        values = sorted(v[-self.n_chars:] for v in values)
        return "".join(values)


class FirstNWords(StringEncoder):

    def __init__(self,
                 n_words: Optional[int] = None,
                 do_sort: bool = False,
                 *args,
                 **kwargs) -> None:
        """
        First N words of the string
        Args:
            n_words:
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.n_words = n_words
        self.do_sort = do_sort

    def _encode(self, value: str) -> str:
        values = list(self.tokenizer(value.lower()))
        if self.do_sort:
            values = sorted(values)
        if self.n_words is not None:
            values = values[: self.n_words]
        return "".join(values)


class NLetterAbbreviation(StringEncoder):

    def __init__(self, n_letters: int = 3, *args, **kwargs):
        """
        First letter of each word of the string sorted alphabetically
        Args:
            n_letters:
            *args:
            **kwargs:
        """
        super().__init__(*args, **kwargs)
        self.n_letters = n_letters

    def _encode(self, value: str) -> str:
        values = self.tokenizer(value.lower())
        values = sorted(v[:1] for v in values)
        return "".join(values[: self.n_letters])
