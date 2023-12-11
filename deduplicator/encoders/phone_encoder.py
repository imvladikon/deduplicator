#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from deduplicator.normalizers.base_normalizer import BaseNormalizer
from typing import Callable, Optional

from deduplicator.encoders.base_encoder import BaseEncoder


class PhoneNormalizer(BaseNormalizer):

    def __init__(self, max_phone_length: int = 10) -> None:
        """
        a naive phone normalizer by removing all non-digit characters and
        padding the phone number with zeros to the left to make it 10 digits
        :param max_phone_length: the maximum length of a phone number
        https://en.wikipedia.org/wiki/National_conventions_for_writing_telephone_numbers
        """
        super(PhoneNormalizer, self).__init__()
        self._normalize = self._normalize
        self.max_phone_length = max_phone_length

    def _normalize(self, phone: str) -> str:
        phone_digits = ''.join(c for c in phone if c.isdigit())
        phone = str(phone_digits).zfill(self.max_phone_length)
        return phone

    def __call__(self, phone: str) -> str:
        return self._normalize(phone)


class PhoneEncoder(BaseEncoder):
    normalizer_class = PhoneNormalizer
    normalizer = None

    def __init__(self,
                 phone_normalizer_service: Optional[Callable] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if phone_normalizer_service is not None:
            self.phone_normalizer = phone_normalizer_service
        elif self.__class__.normalizer is not None:
            self.phone_normalizer = self.__class__.normalizer
        else:
            self.__class__.normalizer = self.normalizer_class()
            self.phone_normalizer = self.__class__.normalizer

    def _encode(self, value: str) -> str:
        if not value:
            return self.empty_value()
        return self.phone_normalizer(value)

    def empty_value(self) -> str:
        return ""


class PhoneNLeftDigitsEncoder(PhoneEncoder):

    def __init__(self, n_left: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n_left

    def _encode(self, value: str) -> str:
        value = super()._encode(value)
        return value[: self.n]


class PhoneNRightDigitsEncoder(PhoneEncoder):

    def __init__(self, n_right: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n_right

    def _encode(self, value: str) -> str:
        value = super()._encode(value)
        return value[-self.n:]
