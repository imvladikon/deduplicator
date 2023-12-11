#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from deduplicator.blockings.rules.blocking_rule_base import BlockingRuleBase


def connected_components(graph):
    return graph.clusters().membership


class GroupByConnectedComponents(BlockingRuleBase):

    def __init__(self, rule, *args, **kwargs):
        """
        Rule that returns the connected components(CC) of the graph of the rule
        CC is referred as transitive closure considers two entities in
        the same cluster if there is a path between them.
        The algorithm does not remove any edges from the input graph.
        Therefore, it results in the highest recall with the cost of very low precision.
        """
        super().__init__(rule, *args, **kwargs)
        self.rule = rule

    def fit(self, df):
        self._groups = connected_components(self.rule.fit(df).graph)
        self._update_graph = True
