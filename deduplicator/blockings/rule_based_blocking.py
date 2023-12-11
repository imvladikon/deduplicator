#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence, Optional

import pandas as pd
from tqdm import tqdm

from deduplicator.blockings.base_blocking import BaseBlocking
from deduplicator.blockings.rules.blocking_rule_base import BlockingRuleBase


class RuleBasedBlocking(BaseBlocking):

    def __init__(
            self,
            blocking_attributes=[],
            blocking_rule: Optional[BlockingRuleBase] = None,
            blocking_filters=None,
            blocking_splitter=None,
            **kwargs
    ):
        super().__init__(blocking_attributes, **kwargs)

        if blocking_rule:
            self.rule = blocking_rule
        elif blocking_attributes:
            self.rule = BlockingRuleBase(*blocking_attributes)
        else:
            raise ValueError(
                "Either blocking_attributes" " or blocking_rule must be provided"
            )
        self._data = None
        self._num_blocks = 0
        self.blocking_filters = blocking_filters
        self.blocking_splitter = blocking_splitter

    def fit(self, records: Sequence[dict], **kwargs) -> "RuleBasedBlocking":
        data = []
        for record in records:
            item = {}
            for k, v in record.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        item[f"{k}.{kk}"] = vv
                else:
                    item[k] = v
            data.append(item)
        if not data:
            raise ValueError("No data to fit")
        data = pd.DataFrame(data)
        self.rule.fit(data)
        data["blocking_id"] = self.rule.groups
        self._data = data
        self._num_blocks = len(data["blocking_id"].unique())
        return self

    def _filter_block(self, block):
        if self.blocking_filters is None:
            return False
        else:
            return any(f(block) for f in self.blocking_filters)

    def _split_block(self, block_id, block):
        if self.blocking_splitter is None:
            yield block_id, block
        else:
            if isinstance(block, list):
                yield from self.blocking_splitter(block_id, block)
            elif isinstance(block, pd.DataFrame):
                yield from self.blocking_splitter(block_id, block.to_dict(orient="records"))

    def iter_blocks(self, verbose=False, **kwargs):
        groups = self._data.groupby("blocking_id")
        pbar = groups if not verbose else tqdm(groups, total=len(groups))
        for group_id, group in pbar:
            for sub_group_id, sub_group in self._split_block(group_id, group):
                if self._filter_block(sub_group):
                    continue
                if isinstance(sub_group, list):
                    yield sub_group_id, sub_group
                elif isinstance(sub_group, pd.DataFrame):
                    yield sub_group_id, sub_group.to_dict(orient="records")

    @property
    def num_blocks(self) -> int:
        return self._num_blocks
