#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from deduplicator.blockings.cartesian_blocking import CartesianBlocking
from deduplicator.blockings.rule_based_blocking import RuleBasedBlocking
from deduplicator.blockings.pruners.sorted_neighbourhood_splitter import SortedNeighbourhoodBlockSplitter
from deduplicator.blockings.pruners.cardinality_filtering import CardinalityBlockingFilter
