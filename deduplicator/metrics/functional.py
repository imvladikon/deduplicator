#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from collections import Counter
from typing import Any, Sequence, Dict, Tuple, List, Optional

import numpy as np
import sklearn.metrics as sk_metrics
from scipy.special import comb as combinations

from deduplicator.utils import LinkedPair, ClusterId


def max_possible_comparison(n_records: int) -> float:
    """
    Returns the maximum number of comparisons between records (~ cartesian product).

    """
    return n_records * (n_records - 1) / 2


def reduction_ratio(*,
                    operations_before_blocking: Optional[int],
                    operations_after_blocking: Optional[int],
                    **kwargs: Any) -> float:
    """
    Returns the ratio of reduced comparisons when blocking is applied.
    it measures how many operations we skip.
    RR = 1 - (n_comparisons after blocking) / (n_comparisons before blocking),
    where
    number of comparisons before blocking = n_records * (n_records - 1) / 2
    """
    if operations_after_blocking is None or operations_before_blocking is None:
        return 0
    if operations_before_blocking == 0:
        return 0
    else:
        return (1 - operations_after_blocking / operations_before_blocking) * 100


def comparison_efficiency(*,
                          operations_before_blocking: Optional[int],
                          operations_after_blocking: Optional[int],
                          **kwargs: Any) -> float:
    """
    Returns the ratio of reduced comparisons when blocking is applied.
    The same as reduction_ratio but in the range [0, inf], where 0 means no reduction and inf means all comparisons are skipped.
    CE = (number of comparisons before blocking) / (number of comparisons after blocking),
    Higher is better and metric is more readable than reduction_ratio (e.g. 30 means 30 times faster than without blocking).
    Args:
        operations_before_blocking:
        operations_after_blocking:
        **kwargs:
    """ # noqa
    if operations_after_blocking is None or operations_before_blocking is None:
        return 1
    if operations_after_blocking == 0:
        return math.inf
    else:
        return operations_before_blocking / operations_after_blocking


def precision(*, true_labels, pred_labels) -> float:
    _, tp_cluster_sizes = np.unique((pred_labels, true_labels), axis=1,
                                    return_counts=True)
    _, p_cluster_sizes = np.unique(pred_labels, return_counts=True)

    tp = np.sum(combinations(tp_cluster_sizes, 2))
    p = np.sum(combinations(p_cluster_sizes, 2))

    del tp_cluster_sizes, p_cluster_sizes
    if p == 0:
        return 100.0
    else:
        return float(100 * tp / p)


def recall(*, true_labels, pred_labels) -> float:
    """
    recall is precision with swapped labels and predictions
    because, recall = TP / (TP + FN) and precision = TP / (TP + FP),
    so when we swap labels and predictions,  FP becomes FN and vice versa.
    """
    return float(precision(true_labels=pred_labels, pred_labels=true_labels))


def pair_completeness(*, true_labels, pred_labels) -> float:
    """Pair Completeness(PC) or 1-Recall which assesses the portion of the duplicate entities that co-occur at least once in Block)""" # noqa
    return 100 - float(recall(true_labels=true_labels, pred_labels=pred_labels))


def pair_quality(*, true_labels, pred_labels) -> float:
    """
    Pairs Quality(PQ) corresponds to precision, as it estimates the portion of non-redundant comparisons that involve matching entities
    """ # noqa
    return float(precision(true_labels=true_labels, pred_labels=pred_labels))


def f1_score(*, true_labels, pred_labels, tol: float = 1e-6) -> float:
    p = precision(true_labels=true_labels, pred_labels=pred_labels)
    r = recall(true_labels=true_labels, pred_labels=pred_labels)
    return float(2 * p * r / (p + r + tol))


def confusion_matrix(*, true_labels, pred_labels) -> Dict[str, int]:
    """
    https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    TP decision assigns two similar entities to the same cluster,
    TN decision assigns two dissimilar entities to different clusters.

    FP decision assigns two dissimilar entities to the same cluster.
    FN decision assigns two similar entities to different clusters.
    """
    _, tp_cluster_sizes = np.unique((pred_labels, true_labels), axis=1,
                                    return_counts=True)
    tp = np.sum(combinations(tp_cluster_sizes, 2))

    _, p_cluster_sizes = np.unique(pred_labels, return_counts=True)
    p = np.sum(combinations(p_cluster_sizes, 2))

    _, t_cluster_sizes = np.unique(true_labels, return_counts=True)
    t = np.sum(combinations(t_cluster_sizes, 2))

    fp = p - tp
    fn = t - tp
    # TN is all possible pairs minus TP, FP, FN
    tn = combinations(len(pred_labels), 2) - p - fn

    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def homogeneity(*, true_labels, pred_labels) -> float:
    """
    A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class.
    https://github.com/Valires/er-evaluation/blob/main/er_evaluation/metrics/_metrics.py#L450
    """ # noqa
    return float(100 * sk_metrics.homogeneity_score(true_labels, pred_labels))


def completeness(*, true_labels, pred_labels) -> float:
    """
    A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class.
    https://github.com/Valires/er-evaluation/blob/main/er_evaluation/metrics/_metrics.py#L469
    """ # noqa
    return float(100 * sk_metrics.completeness_score(true_labels, pred_labels))


def v_measure(*, true_labels, pred_labels, beta=1.0) -> float:
    """
    The V-measure is the harmonic mean between homogeneity and completeness:
    """
    return float(100 * sk_metrics.v_measure_score(true_labels, pred_labels, beta=beta))


def rand_index(*, true_labels, pred_labels) -> float:
    """
    accuracy of the clustering
    """
    return float(100 * sk_metrics.rand_score(true_labels, pred_labels))


def adjusted_random_index(*, true_labels, pred_labels) -> float:
    """The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and true
    clusterings.

        ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

    >> adjusted_rand_index([0, 0, 1, 2], [0, 0, 1, 1])
      57.0
    >> adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0])
      100.0
    """
    assert (-1 not in pred_labels) and (-1 not in true_labels), "-1 found in labels"
    return float(sk_metrics.adjusted_rand_score(true_labels, pred_labels) * 100)


def precision_recall_f1_from_links(true_links: Sequence[LinkedPair],
                                   pred_links: Sequence[LinkedPair],
                                   etol: float = 1e-10) -> Tuple[float, float, float]:
    """Computes precision recall and F1-score given true and links
    Args:
        true_links: a list of pairs of linked elements
            for instance [(1, 0), (3, 0), (3, 1)]
        pred_links: a list of pairs of prediceted links
            for instance [(1, 0), (3, 1)]

    Returns:
        precision, recall, f1-score
    """
    st = set(true_links)
    sp = set(pred_links)
    if not sp or not st:
        return 0, 0, 0

    tp = len(st & sp)

    precision = tp / len(sp) * 100
    recall = tp / len(st) * 100
    f1 = 2 * precision * recall / (precision + recall + etol)

    return precision, recall, f1


def calc_cluster_statistics(labels: List[ClusterId]) -> Tuple[int, int]:
    counter = Counter(labels)
    num_non_singleton_clusters = sum(1 for _ in counter if counter[_] > 1)
    num_clusters = len(counter)
    return num_non_singleton_clusters, num_clusters
