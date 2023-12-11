#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pprint
import uuid
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache, reduce
from types import MappingProxyType
from typing import Dict, Any, List, Tuple, Union, Callable

import numpy as np

ClusterId = int
MentionId = int
LinkedPair = Tuple[MentionId, MentionId]
_ENTID_TO_CLUST = Dict[MentionId, ClusterId]
_ENTID_TO_PROPS = Dict[MentionId, Dict]

FROZEN_EMPTY_DICT = MappingProxyType({})

ENTITY_TYPE = 'entity_type'
SOURCE_ID = 'source_id'
MENTION_ID = 'id'
CLUSTER_ID = 'cluster_id'


def pformat(obj: Any, num_chars: int = 1000, compact: bool = False) -> str:
    if compact:
        obj_str = pprint.pformat(obj, compact=True, depth=2).replace('\n', ' ')
    else:
        obj_str = pprint.pformat(obj)
    if len(obj_str) > num_chars:
        obj_str = obj_str[:num_chars] + ' ...'
    return obj_str


def flatten_dict(d: Dict[str, Any], sep: str = '||') -> Dict[str, Any]:
    to_process = [(tuple(), k, v) for k, v in d.items()]
    processed = {}
    while len(to_process) > 0:
        p, k, v = to_process.pop()
        if not isinstance(v, dict):
            processed[p + (k,)] = v
        else:
            prefix = p + (k,)
            to_process.extend([(prefix, key, val) for key, val in v.items()])
    return {sep.join(k): v for k, v in processed.items()}


def get_flat_schema(mentions: List[Dict[str, Any]], sep: str = '||') -> Dict[str, type]:
    flat_schema = {}
    for m in mentions:
        flat_mention = flatten_dict(m, sep)
        mention_schema = {k: {'type': type(v).__name__} for k, v in flat_mention.items()}
        flat_schema.update(mention_schema)
    return flat_schema


def labels_to_pairs(labels: List[ClusterId]) -> List[LinkedPair]:
    """Computes all pairs for a given cluster labels list

    For instance for the cluster indices list [0, 0, 1, 0]
    the result will be: [(1, 0), (3, 0), (3, 1)]

    Args:
        labels: a list of cluster indices

    Returns:
        pairs: list of pair tuples of linked indices
            NOTE that only one link direction is returned
    """
    assert -1 not in labels, "-1 found in labels"
    pairs = []
    clusters_dict = {}
    for node, label in enumerate(labels):
        for linked_node in clusters_dict.get(label, []):
            pairs.append((node, linked_node))
        clusters_dict.setdefault(label, []).append(node)
    return pairs


def pairs_to_labels(pairs: List[LinkedPair],
                    total_mentions: int,
                    drop_singletons: bool = False) -> List[ClusterId]:
    """Converts linked pairs to list of cluster indices

    Args:
        pairs: a list of pairs of linked mentions, for instance
            [(1, 0), (3, 0), (3, 1)]
            Pairs should support transitivity law, otherwise correctness is not
            guaranted.
        total_mentions: total number of mentions including not linked ones
        drop_singletons: whether to assign -1 to all singleton mentions

    Returns:
        labels: a list of cluster indices, for instance [0, 0, 1, 0]
    """
    # each node links itself
    node_matches = [{i} for i in range(total_mentions)]
    # updating node's linked nodes based on linked pairs
    for pair in pairs:
        node_a, node_b = pair
        node_matches[node_a].update(node_matches[node_b])
        node_matches[node_b] = node_matches[node_a]

    final_labels = [-1 for _ in range(total_mentions)]
    cluster_idx = 0
    for node, linked_nodes in enumerate(node_matches):
        if final_labels[node] > -1:  # Already assigned to a cluster
            continue
        if drop_singletons and (len(linked_nodes) == 1):  # Linked to itself only
            continue
        for l_node in linked_nodes:  # Propogating label to linked nodes
            final_labels[l_node] = cluster_idx
        cluster_idx += 1
    return final_labels


def clusters_to_labels(clusters: List[Dict],
                       total_mentions: int,
                       drop_singletons: bool = False) -> List[ClusterId]:
    """Converts clusters to labels

    Args:
        clusters: a list of cluster dictionaries, each dict has mention_id as a key
            and a special CLUSTER_ID key. Each cluster should have at least 2 elements.
                [ {0: None, 1: None, 3: None, "cluster_id": "cluster0"} ]
        total_mentions: total number of mentions including not linked ones
        drop_singletons: whether to assign -1 to all singleton mentions

    Returns:
        labels: a list of cluster indices, for instance [0, 0, 1, 2, 0]
    """
    labels = [-1 for _ in range(total_mentions)]
    for cluster_idx, mention_ids in enumerate(clusters):
        for node in mention_ids:
            if node == CLUSTER_ID:
                continue
            # TODO: assuming that mention id is convertable to int, and starts from 0
            node = int(node)
            if labels[node] > -1:  # Already assigned to a cluster
                raise ValueError(f'mention {node} is present in two clusters')
            labels[node] = cluster_idx

    if not drop_singletons:
        cluster_idx = len(clusters)
        for node, node_label in enumerate(labels):
            if node_label == -1:
                labels[node] = cluster_idx
                cluster_idx += 1
    return labels


def new_id() -> str:
    return str(uuid.uuid4())


def wrap_in_list(value: Any) -> List[Any]:
    if not isinstance(value, (tuple, list)):
        value = [value]
    return value


def apply_comparator(group_a: Union[Any, List[Any]],
                     group_b: Union[Any, List[Any]],
                     comparator: Callable,
                     cache_size: int = 10) -> float:
    comparator = lru_cache(maxsize=cache_size)(comparator)

    group_a = [str(a) for a in wrap_in_list(group_a) if a]
    group_b = [str(b) for b in wrap_in_list(group_b) if b]
    if not group_a or not group_b:
        return 0.0
    if group_a == group_b:
        return 1.0
    return float(
        np.mean(
            [
                comparator(a, b) if len(a) < len(b) else comparator(b, a)
                for a in group_a
                for b in group_b
            ]
        )
    )


def dataframe_from_mentions(mentions: List[Dict],
                            # flat_schema: Dict[str, type],
                            sep: str = '||') -> dict:
    import pandas as pd

    mention_types = set(m['entity_type'] for m in mentions)
    dataframes_dict = {}
    for mention_type in mention_types:
        current_mentions = [m for m in mentions if m['entity_type'] == mention_type]
        flat_schema = get_flat_schema(current_mentions)
        data_dict = {k: [] for k in flat_schema}
        for m in current_mentions:
            for attr in flat_schema:
                nested_keys = attr.split(sep)
                try:
                    data_dict[attr].append(reduce(dict.get, nested_keys, m))
                except TypeError:
                    data_dict[attr].append(None)
        dataframes_dict[mention_type] = pd.DataFrame(data_dict)
    return dataframes_dict


def clusters_to_indexes(clusters: List[Dict]) -> Tuple[_ENTID_TO_CLUST, _ENTID_TO_PROPS]:
    """
    converts list of clusters to two dictionaries:
    1. entid_to_clustid: maps entity id to cluster id (guid)
    2. entid_to_props: maps entity id to its properties (empty dict if no properties)
    Args:
        clusters: list of clusters:
        [{130: {}, 861: {},
        'cluster_id': 'c675c8c5-c448-4c15-859a-400f62f41d41', ...}]
    Returns:
        entid_to_clustid: {108: 'c675c8c5-c448-4c15-859a-400f62f41d41', ...}
        entid_to_props: {108: {}, ...}
    """
    entid_to_clustid = {}
    entid_to_props = {}
    for clust in clusters:
        clust = deepcopy(clust)
        clust_id = clust.pop('cluster_id')
        entid_to_clustid.update({ent_id: clust_id for ent_id in clust})
        entid_to_props.update({ent_id: clust[ent_id] for ent_id in clust})
    return entid_to_clustid, entid_to_props


def merge_clusterizations(clusters_1: List[Dict], clusters_2: List[Dict]) -> List[Dict]:
    """
    merges two clusterizations into one
    Args:
        clusters_1: [{130: {}, 861: {}, 'cluster_id': 'c675c8c5-c448-4c15-859a-400f62f41d41', ...}] # noqa E501
        clusters_2: [{138: {}, 861: {}, 'cluster_id': 'c675c8c5-c448-4c15-859a-400f62f41d42', ...}] # noqa E501
    Returns:
        merged_clusters: [{130: {}, 861: {}, 862: {}, 'cluster_id': 'c675c8c5-c448-4c15-859a-400f62f41d41', ...}] # noqa E501
    """
    ent2clust_sm, entid_to_prop = clusters_to_indexes(clusters_1)
    ent2clust_id, entid_to_prop_id = clusters_to_indexes(clusters_2)
    entid_to_prop.update(entid_to_prop_id)
    clust2merge_id = {}
    merge_id2clust = defaultdict(list)
    all_entity_ids = ent2clust_sm.keys() | ent2clust_id.keys()
    for ent_id in all_entity_ids:
        id_clust = ent2clust_id.get(ent_id)
        sm_clust = ent2clust_sm.get(ent_id)
        if id_clust and sm_clust:
            sm_mid = clust2merge_id.get(sm_clust)
            id_mid = clust2merge_id.get(sm_clust)
            if sm_mid and id_mid and sm_mid == id_mid:
                current_id = sm_mid
            elif sm_mid and id_mid and sm_mid != id_mid:
                sm_ids = merge_id2clust.pop(sm_mid)
                merge_id2clust[id_mid].extend(sm_ids)
                for sm_id in sm_ids:
                    clust2merge_id[sm_id] = id_mid
                current_id = id_mid
            elif sm_mid or id_mid:
                current_id = sm_mid or id_mid
            else:
                current_id = id_clust
            clust2merge_id[sm_clust] = current_id
            clust2merge_id[id_clust] = current_id
            merge_id2clust[current_id].extend([id_clust, sm_clust])

    clust_dict = defaultdict(dict)
    for ent_id in all_entity_ids:
        id_clust = ent2clust_id.get(ent_id)
        sm_clust = ent2clust_sm.get(ent_id)
        clust = id_clust or sm_clust
        cluste_id = clust2merge_id.get(clust, clust)
        clust_dict[cluste_id][ent_id] = entid_to_prop[ent_id]
    clusters = []
    for clust_id, ents in clust_dict.items():
        ents['cluster_id'] = clust_id
        clusters.append(ents)
    return clusters
