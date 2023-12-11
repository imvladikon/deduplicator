#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import combinations

import igraph as ig

from deduplicator.blockings.rules.blocking_rule_base import BlockingRuleBase
from deduplicator.blockings.rules.group_by_exact_rules import ExactGroupBy


class GroupByLogicalAND(BlockingRuleBase):
    """
    the logical conjunction
    for blocking rules or intersection of the blocking rules (graphs)
    """

    def fit(self, df):
        if self.level == "groups":
            self._update_graph = True
            self._groups = BlockingRuleBase.groups_from_rules(self.rules, df)
        else:
            self._update_groups = True
            self._graph = ig.intersection([rule.get_graph(df) for rule in self.rules])
        return self


class GroupByLogicalOR(BlockingRuleBase):
    """
    the logical disjunction for blocking rules or union of the blocking rules (graphs)
    """

    def fit(self, df):
        if self.level == "groups":
            graphs_vect = [rule.get_path_graph(df) for rule in self.rules]
            self._graph = ig.union(graphs_vect)
        else:
            self._graph = ig.union([rule.get_graph(df) for rule in self.rules])
        self._update_groups = True
        return self


class GroupByCombinationsExceptK(GroupByLogicalOR):

    def __init__(self, *rules, k: int = 0, level="groups"):
        rules = [
            ExactGroupBy(*x, level=level) for x in combinations(rules, len(rules) - k)
        ]
        super().__init__(*rules)
