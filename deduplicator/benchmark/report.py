#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from pathlib import Path
from collections import MutableMapping
from typing import Dict, List, Union, Optional, Any

from deduplicator import utils
from deduplicator.metrics import (ClusterMetrics,
                                     ConfusionMatrix,
                                     calc_cluster_statistics,
                                     reduction_ratio,
                                     comparison_efficiency)

logger = utils.get_logger(__name__)


class ReportBuilder(MutableMapping, dict):

    def __init__(self,
                 true_labels: List[utils.ClusterId],
                 pred_labels: List[utils.ClusterId],
                 block_labels: Optional[List[utils.ClusterId]] = None,
                 block_stats: Optional[Dict] = None,
                 **kwargs) -> None:
        """
        Calculated metrics based on input true and predicted clusters.
        """
        metrics = ClusterMetrics()(true_labels=true_labels, pred_labels=pred_labels)
        confusion_matrix = ConfusionMatrix()(true_labels=true_labels,
                                             pred_labels=pred_labels)

        num_ntc, _ = calc_cluster_statistics(true_labels)
        num_npc, _ = calc_cluster_statistics(pred_labels)
        self._report = {
            'AdjustedRandomIndex': metrics['adjusted_random_index'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-measure': metrics['f1_score'],
            'Completeness': metrics['completeness'],
            'Homogeneity': metrics['homogeneity'],
            'V-measure': metrics['v_measure'],
            'RandIndex': metrics['rand_index'],
            'TP': confusion_matrix['TP'],
            'FP': confusion_matrix['FP'],
            'FN': confusion_matrix['FN'],
            'TN': confusion_matrix['TN'],
            'NumPredictedPairs': confusion_matrix['TP'] + confusion_matrix['FP'],
            'NumTruePairs': confusion_matrix['TP'] + confusion_matrix['FN'],
            'NumPredictedNonSingletonClusters': num_npc,
            'NumTrueNonSingletonClusters': num_ntc
        }
        self._update_blocking_metrics(true_labels, block_labels, block_stats)

    @property
    def report(self) -> Dict:
        return self._report

    def print(self, startswith: str = "", f: Optional[io.StringIO] = None) -> None:
        for param_name, value in self.report.items():
            if param_name.startswith(startswith):
                if f is not None:
                    f.write(f"{param_name}: {value:.2f}\n")
                else:
                    logger.info(f"{param_name}: {value:.2f}")

    @classmethod
    def from_clusters(cls,
                      true_clusters: List[Dict],
                      pred_clusters: List[Dict],
                      total_mentions: int,
                      block_labels: Optional[List[utils.ClusterId]] = None,
                      block_stats: Optional[Dict] = None,
                      **kwargs) -> "ReportBuilder":
        true_labels = utils.clusters_to_labels(true_clusters, total_mentions)
        pred_labels = utils.clusters_to_labels(pred_clusters, total_mentions)
        return cls.from_labels(true_labels,
                               pred_labels,
                               block_labels,
                               block_stats,
                               **kwargs)

    @classmethod
    def from_labels(cls,
                    true_labels: List[utils.ClusterId],
                    pred_labels: List[utils.ClusterId],
                    block_labels: Optional[List[utils.ClusterId]] = None,
                    block_stats: Optional[Dict] = None,
                    **kwargs) -> "ReportBuilder":
        return cls(true_labels, pred_labels, block_labels, block_stats, **kwargs)

    @classmethod
    def from_pairs(cls,
                   true_pairs: List[utils.LinkedPair],
                   pred_pairs: List[utils.LinkedPair],
                   total_mentions: int,
                   **kwargs) -> "ReportBuilder":
        true_labels = utils.pairs_to_labels(true_pairs, total_mentions)
        pred_labels = utils.pairs_to_labels(pred_pairs, total_mentions)
        return cls(true_labels, pred_labels, **kwargs)

    @classmethod
    def from_path(cls, path: Union[Path, str]) -> "ReportBuilder":
        # TODO: read from parquet
        # import pandas as pd
        # pred_clusters = pd.read_parquet(str(path), columns=['id', 'firstname'])
        raise NotImplementedError

    def _update_blocking_metrics(self,
                                 true_labels,
                                 block_labels=None,
                                 block_stats=None) -> None:
        # TODO: add fair metrics calculation in the overlapping case
        if block_stats is None or block_labels is None:
            return

        operations_before_blocking = block_stats['operations_before_blocking']
        operations_after_blocking = block_stats['operations_after_blocking']

        reduction_ratio_score = reduction_ratio(
            operations_after_blocking=operations_after_blocking,
            operations_before_blocking=operations_before_blocking)
        comparison_efficiency_ratio = comparison_efficiency(
            operations_after_blocking=operations_after_blocking,
            operations_before_blocking=operations_before_blocking)

        cluster_metrics = ClusterMetrics()(true_labels=true_labels,
                                           pred_labels=block_labels)
        self._report.update({
            'OperationsBeforeBlocking': block_stats['operations_before_blocking'],
            'OperationsAfterBlocking': block_stats['operations_after_blocking'],
            'BlockingPrecision': cluster_metrics['precision'],
            'BlockingRecall': cluster_metrics['recall'],
            'BlockingF1-measure': cluster_metrics['f1_score'],
            'BlockingCompleteness': cluster_metrics['completeness'],
            'BlockingHomogeneity': cluster_metrics['homogeneity'],
            'BlockingV-measure': cluster_metrics['v_measure'],
            'BlockingRandIndex': cluster_metrics['rand_index'],
            'BlockingAdjustedRandIndex': cluster_metrics['adjusted_random_index'],
            'ReductionRatio': reduction_ratio_score,
            'ComparisonEfficiency': comparison_efficiency_ratio,
        })

    def __getitem__(self, key: str) -> float:
        return self.report[key]

    def __setitem__(self, key: str, value: Any) -> float:
        self.report[key] = value
        return self.report[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def to_dict(self) -> Dict:
        return dict(self.report)
