#!/usr/bin/env python3
from deduplicator.metrics.functional import (precision,
                                                recall,
                                                f1_score,
                                                completeness,
                                                homogeneity,
                                                adjusted_random_index,
                                                v_measure,
                                                rand_index)
from deduplicator.metrics.metrics_wrapper import MetricsWrapper


class ClusterMetrics(MetricsWrapper):

    def __init__(self, **kwargs):
        metrics = {
            "completeness": completeness,
            "homogeneity": homogeneity,
            "adjusted_random_index": adjusted_random_index,
            "rand_index": rand_index,
            "v_measure": v_measure,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
        super().__init__(metrics, **kwargs)
