#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from deduplicator.blockings.pruners.base_blocking_filter import BaseBlockingFilter


class CardinalityBlockingFilter(BaseBlockingFilter):

    def __init__(self, min_block_size: int = 2, max_block_size: int = np.inf):
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size

    def __call__(self, block, *args, **kwargs):
        block_size = len(block)
        return (block_size > self.max_block_size) or (block_size < self.min_block_size)
