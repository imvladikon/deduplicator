#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional, Callable, Union, Any

from deduplicator.encoders.base_encoder import BaseEncoder
from deduplicator.tokenizers import NumericTokenizer


class BaseIntegerEncoder(BaseEncoder):

    def __init__(self, tokenizer: Optional[Union[Callable, str]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tokenizer is None:
            self.tokenizer = lambda x: x
        elif isinstance(tokenizer, str):
            tokenizer_factory = {
                "whitespace": lambda x: x.split(),
                "char": lambda x: list(x),
                "fuzzy": NumericTokenizer(),
            }
            self.tokenizer = tokenizer_factory[tokenizer]
        else:
            self.tokenizer = tokenizer

    def _encode(self, value: int) -> str:
        if not value:
            return self.empty_value()
        values = self.tokenizer(value)
        return "".join(values)

    def empty_value(self) -> str:
        return ""


class RoundInteger(BaseIntegerEncoder):

    def _encode(self, value: int) -> str:
        if not value:
            return self.empty_value()
        return str(round(value))


class SortedIntegers(BaseIntegerEncoder):

    def __init__(self, max_length: Optional[int] = None, *args, **kwargs):
        super().__init__(tokenizer="fuzzy", *args, **kwargs)
        self.max_length = max_length

    def _encode(self, value: Any) -> str:
        if not value:
            return self.empty_value()
        values = self.tokenizer(str(value))
        values = sorted(values)
        if self.max_length is not None:
            values = values[: self.max_length]
        return "".join(values)


class FirstInteger(BaseIntegerEncoder):

    def _encode(self, value: int) -> str:
        if not value:
            return self.empty_value()
        return str(value)[0]


class LastInteger(BaseIntegerEncoder):

    def _encode(self, value: int) -> str:
        if not value:
            return self.empty_value()
        return str(value)[-1]


class LargestInteger(BaseIntegerEncoder):

    def __init__(self, *args, **kwargs):
        super().__init__(tokenizer="fuzzy", *args, **kwargs)

    def _encode(self, value: int) -> str:
        if not value:
            return self.empty_value()
        return str(max(self.tokenizer(str(value))))
