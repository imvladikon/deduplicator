#!/usr/bin/env python3
from typing import Any

from deduplicator.metrics.functional import confusion_matrix
from deduplicator.metrics.metrics_wrapper import MetricsWrapper


class ConfusionMatrix(MetricsWrapper):

    def __init__(self, **kwargs):
        metrics = {
            "confusion_matrix": confusion_matrix
        }
        super().__init__(metrics, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> None:
        for metric_name, metric_fn in self.metrics.items():
            dict_value = metric_fn(*args, **kwargs)
            for key in dict_value:
                self._computed_metrics.setdefault(metric_name, {}).setdefault(key, []).append(dict_value[key]) # noqa E501

    def _metric(self) -> Any:
        ret = {}
        for metric_name in self._computed_metrics:
            for key in self._computed_metrics[metric_name]:
                # confusion_matrix is considered as an additive metric
                ret[key] = sum(self._computed_metrics[metric_name][key])
        return ret
