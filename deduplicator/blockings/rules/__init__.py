#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from deduplicator.blockings.rules.group_by_exact_rules import (
    ExactGroupBy,
    PhoneticGroupBy,
    ConsonantGroupBy,
    DateGroupBy,
    FirstIntegerGroupBy,
    LargestIntegerGroupBy,
    LastIntegerGroupBy,
    SortedIntegersGroupBy,
    RoundIntegerGroupBy,
    FirstNCharsLastWordGroupBy,
    FirstNCharsGroupBy,
    FirstNWordsGroupBy,
    NLetterAbbreviationGroupBy,
    StringEncoderGroupBy,
    LastNCharsGroupBy
)
from deduplicator.blockings.rules.group_by_logical_operations import (
    GroupByLogicalOR,
    GroupByLogicalAND,
    GroupByCombinationsExceptK
)
