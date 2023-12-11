#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence

from deduplicator.tokenizers.base_tokenizer import BaseTokenizer


class WhitespaceSpaceTokenizer(BaseTokenizer):
    """Whitespace tokenizer that splits on spaces."""

    def __init__(self):
        super().__init__()

    def _tokenize(self, text: str) -> Sequence[str]:
        return text.split()