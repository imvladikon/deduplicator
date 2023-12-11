#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional, Callable

from deduplicator.encoders.base_encoder import BaseEncoder
from deduplicator.encoders.phonetic_encoders import PhoneticEncoder, ConsonantEncoder
from deduplicator.encoders.date_encoder import DateEncoder
from deduplicator.encoders.geo_encoder import GeoEncoder
from deduplicator.encoders.integer_encoders import (FirstInteger,
                                                    LastInteger,
                                                    LargestInteger,
                                                    RoundInteger,
                                                    SortedIntegers)
from deduplicator.encoders.string_encoders import (FirstNChars,
                                                   LastNChars,
                                                   FirstNWords,
                                                   StringEncoder,
                                                   NLetterAbbreviation,
                                                   FirstNCharsLastWord)
from deduplicator.blockings.rules.blocking_rule_base import BlockingRuleBase


class ExactGroupBy(BlockingRuleBase):

    def __init__(
            self, *rules, level="groups", encoder: Optional[BaseEncoder] = None
    ) -> None:
        super().__init__(*rules, level=level)
        self.encoder = encoder

    def fit(self, df):
        if self.encoder is not None:
            rules = []
            encoder_name = self.encoder.__class__.__name__.lower()
            for rule in self.rules:
                rule_name = f"__{rule}_{encoder_name}__"
                df[rule_name] = df[rule].apply(self.encoder)
                rules.append(rule_name)
            self.rules = rules
        super().fit(df)
        return self


class PhoneticGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, **kwargs) -> None:
        super().__init__(column_name, encoder=PhoneticEncoder(**kwargs))


class ConsonantGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, **kwargs) -> None:
        super().__init__(column_name, encoder=ConsonantEncoder(**kwargs))


class DateGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, **kwargs) -> None:
        super().__init__(column_name, encoder=DateEncoder(**kwargs))


class GeoGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, **kwargs) -> None:
        super().__init__(column_name, encoder=GeoEncoder(**kwargs))


class FirstIntegerGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, **kwargs) -> None:
        super().__init__(column_name, encoder=FirstInteger(**kwargs))


class LastIntegerGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, **kwargs) -> None:
        super().__init__(column_name, encoder=LastInteger(**kwargs))


class LargestIntegerGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, **kwargs) -> None:
        super().__init__(column_name, encoder=LargestInteger(**kwargs))


class RoundIntegerGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, **kwargs) -> None:
        super().__init__(column_name, encoder=RoundInteger(**kwargs))


class SortedIntegersGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, max_length: int = 3, **kwargs) -> None:
        super().__init__(column_name,
                         encoder=SortedIntegers(max_length=max_length, **kwargs))


class FirstNCharsGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, n_chars: int = 3, **kwargs) -> None:
        super().__init__(column_name, encoder=FirstNChars(n_chars=n_chars, **kwargs))


class LastNCharsGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, n_chars: int = 3, **kwargs) -> None:
        super().__init__(column_name, encoder=LastNChars(n_chars=n_chars, **kwargs))


class FirstNWordsGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, n_words: int = 1, **kwargs) -> None:
        super().__init__(column_name, encoder=FirstNWords(n_words=n_words, **kwargs))


class StringEncoderGroupBy(ExactGroupBy):

    def __init__(self,
                 column_name: str,
                 tokenizer: Optional[Callable] = None,
                 **kwargs) -> None:
        super().__init__(column_name,
                         encoder=StringEncoder(tokenizer=tokenizer, **kwargs))


class NLetterAbbreviationGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, n_letters: int = 3, **kwargs) -> None:
        super().__init__(column_name,
                         encoder=NLetterAbbreviation(n_letters=n_letters, **kwargs))


class FirstNCharsLastWordGroupBy(ExactGroupBy):

    def __init__(self, column_name: str, n_chars: int = 3, **kwargs) -> None:
        super().__init__(column_name,
                         encoder=FirstNCharsLastWord(n_chars=n_chars, **kwargs))
