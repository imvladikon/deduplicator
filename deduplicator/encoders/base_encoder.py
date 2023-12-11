#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC
from typing import Sequence, Optional, Iterable, Union, List

from deduplicator.utils import batched, flatten


class BaseEncoder(ABC):

    def __init__(self, *args, **kwargs):
        """
        base encoder class for blocking, vectorization, encoding, shingling(for LSH), etc.
        Args:
            *args:
            **kwargs:
        """
        pass

    def __call__(self, value, *args, **kwargs):
        return self._encode(value)

    def _encode(self, value: str) -> str:
        raise NotImplementedError("abstract method")

    def encode_batch(
            self,
            values: Sequence[str],
            batch_size: Optional[int] = None,
            num_threads: Optional[int] = None,
            verbose: bool = False,
            return_list: bool = False,
    ) -> Union[Sequence[str], Iterable[List[str]]]:
        if batch_size is None:
            batch_size = 1

        def _it():
            for batch in batched(values, batch_size):
                yield [self(value) for value in batch]

        if return_list:
            return list(flatten(_it()))
        return _it()

    def empty_value(self) -> str:
        raise NotImplementedError("abstract method")
