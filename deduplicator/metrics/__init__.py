#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from deduplicator.metrics.functional import (max_possible_comparison,
                                                reduction_ratio,
                                                comparison_efficiency)
from deduplicator.metrics.cluster_metrics import ClusterMetrics
from deduplicator.metrics.confusion_matrix import ConfusionMatrix
from deduplicator.metrics.functional import calc_cluster_statistics
