#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Callable, Optional

import dateparser
from deduplicator.encoders.base_encoder import BaseEncoder
from datetime import datetime


class DateNormalizerService:

    def __call__(self, value: str) -> Optional[datetime]:
        return dateparser.parse(value)


class DateEncoder(BaseEncoder):
    normalizer_class = DateNormalizerService
    normalizer = None

    def __init__(self,
                 date_normalizer_service: Optional[Callable] = None,
                 format_string: str = "%Y",
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if date_normalizer_service is not None:
            self.date_normalizer = date_normalizer_service
        elif self.__class__.normalizer is not None:
            self.date_normalizer = self.__class__.normalizer
        else:
            self.__class__.normalizer = self.normalizer_class()
            self.date_normalizer = self.__class__.normalizer
        self.format_string = format_string

    def _encode(self, value: str) -> str:
        if not value:
            return self.empty_value()
        dt = self.date_normalizer(value)
        if dt is None:
            return self.empty_value()
        return datetime(year=dt.year, month=dt.month, day=dt.month).strftime(
            self.format_string
        )

    def empty_value(self) -> str:
        return "0000"


class YearEncoder(DateEncoder):

    def _encode(self, value: str) -> str:
        if not value:
            return self.empty_value()
        dt = self.date_normalizer(value)
        if dt is None:
            return self.empty_value()
        return str(dt.year)

    def empty_value(self) -> str:
        return "0000"


class MonthEncoder(DateEncoder):

    def _encode(self, value: str) -> str:
        if not value:
            return self.empty_value()
        dt = self.date_normalizer(value)
        if dt is None:
            return self.empty_value()
        return str(dt.month)

    def empty_value(self) -> str:
        return "00"


class DayEncoder(DateEncoder):

    def _encode(self, value: str) -> str:
        if not value:
            return self.empty_value()
        dt = self.date_normalizer(value)
        if dt is None:
            return self.empty_value()
        return str(dt.day)

    def empty_value(self) -> str:
        return "00"
