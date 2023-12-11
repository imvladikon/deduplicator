#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Callable, Dict

import numpy as np

from deduplicator.metrics.base_metric import BaseMetric


class MetricsWrapper(BaseMetric):

    def __init__(self, metrics: Dict[str, Callable], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.metrics = metrics
        self._computed_metrics = {}

    def update(self, *args: Any, **kwargs: Any) -> None:
        for metric_name, metric_fn in self.metrics.items():
            self._computed_metrics.setdefault(metric_name, []).append(
                metric_fn(*args, **kwargs))

    def _metric(self) -> Any:
        ret = {}
        for metric_name in self._computed_metrics:
            ret[metric_name] = float(np.mean(self._computed_metrics[metric_name]))
        return ret

    def reset(self) -> None:
        self._computed_metrics = {}
