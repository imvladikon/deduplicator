#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    higher_is_better: bool = True

    def __init__(self, *args, **kwargs):
        pass

    def compute(self):
        return self._metric()

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def _metric(self) -> Any:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.update(*args, **kwargs)
        return self.compute()
