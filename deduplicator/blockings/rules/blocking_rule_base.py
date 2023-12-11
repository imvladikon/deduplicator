#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
from abc import ABC
from typing import Union

import numpy as np
import pandas as pd
import igraph as ig

from deduplicator.utils import iter_pairwise, flatten


class BlockingRuleBase(ABC):

    def __init__(self,
                 *rules: Union[str, "BlockingRuleBase"],
                 level: str = "groups") -> None:
        """
        Base class for aggregation of the results using simple transitivity logic
        Args:
            *rules: list of rules either as strings or BlockingRuleBase objects.
            level: "groups" or "graph".
            if logical and should be done after resolving clusters ("groups")
            or at the blocking graph level ("graph")
        """
        self.rules = rules

        self.level = level

        self._graph = None
        self._groups = None

        self._update_graph = False
        self._update_groups = False

    def fit(self, df):
        """
        default implementation of fit method is
        conjunction of the rules(intersection of the graphs)
        """
        if self.level == "groups":
            self._update_graph = True
            self._groups = BlockingRuleBase.groups_from_rules(self.rules, df)
        else:
            self._update_groups = True
            self._graph = ig.intersection([rule.get_graph(df) for rule in self.rules])
        return self

    @property
    def graph(self) -> ig.Graph:
        if self._update_graph:
            self._graph = BlockingRuleBase.graph_from_groups(self._groups)
            self._update_graph = False
        return self._graph

    @property
    def groups(self):
        if self._update_groups:
            self._groups = np.array(self._graph.clusters().membership)
            self._update_groups = False
        return self._groups

    def group_by(self, df: pd.DataFrame):
        self.fit(df)
        return df.groupby(self.groups)

    def get_graph(self, df: pd.DataFrame):
        return self.fit(df).graph

    @staticmethod
    def make_groups(rule, df: pd.DataFrame):
        """
        Factorizing a column of a dataframe and returning an integer vector U such that U[i] indicates
        the cluster to which row i belongs (ordinal encoding)
        NaN values are considered to be non-matching.
        Args:
            rule: string or BlockingRuleBase
            df: pandas Dataframe to which the rule is fitted.
        Returns:
            vector of cluster indices (ordinal encoding)
        Examples:

        >>> import pandas as pd
        >>> df = pd.DataFrame({"fname":["John", "John", "Kevin"], "lname":["Doe", "Dowson", pd.NA]})
        >>> BlockingRuleBase.make_groups("fname", df)
        [0 0 1]
        """  # noqa
        if isinstance(rule, str):
            codes, _ = pd.factorize(df[rule])
            codes = np.array(codes, dtype=np.int32)
            nan_indices = codes == -1
            # filling each NaN in ordinal encoding
            # with a unique integer with offset of used codes
            codes[nan_indices] = np.arange(len(codes), len(codes) + sum(nan_indices))
            return codes
        elif isinstance(rule, BlockingRuleBase):
            return rule.fit(df).groups

    def get_path_graph(self, df: pd.DataFrame) -> ig.Graph:
        """
        Getting graph path that corresponding to the clustering of rule (cluster elements are connected as a path)
        Args:
            df: pandas Dataframe to which the blocking rule is fitted.
        Returns:
            Graph where nodes according to the rule in the same cluster (connected as graph paths)
        """  # noqa
        groups = BlockingRuleBase.make_groups(self, df)
        clust = pd.DataFrame({"groups": groups}).groupby("groups").indices
        graph = ig.Graph(n=df.shape[0])
        graph.add_edges(flatten(iter_pairwise(c) for c in clust.values()))
        return graph

    @staticmethod
    def groups_from_rules(rules, df: pd.DataFrame):
        arr = np.array([BlockingRuleBase.make_groups(rule, df) for rule in rules]).T
        groups = np.unique(arr, axis=0, return_inverse=True)[1]
        return groups

    @staticmethod
    def graph_from_groups(groups):
        clust = pd.DataFrame({"groups": groups}).groupby("groups").indices
        graph = ig.Graph(n=len(groups))
        graph.add_edges(itertools.chain.from_iterable(itertools.combinations(c, 2)
                                                      for c in clust.values()))
        return graph

    def __and__(self, other: "BlockingRuleBase"):
        from deduplicator.blockings.rules import GroupByLogicalAND

        return GroupByLogicalAND(self, other)

    def __or__(self, other: "BlockingRuleBase"):
        from deduplicator.blockings.rules import GroupByLogicalOR

        return GroupByLogicalOR(self, other)

    def __rand__(self, other: "BlockingRuleBase"):
        from deduplicator.blockings.rules import GroupByLogicalAND

        return GroupByLogicalAND(self, other)

    def __ror__(self, other: "BlockingRuleBase"):
        from deduplicator.blockings.rules import GroupByLogicalOR

        return GroupByLogicalOR(self, other)
