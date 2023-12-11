#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence, Optional

from deduplicator.utils.iter_utils import sliding_window


class SortedNeighbourhoodBlockSplitter:

    def __init__(self,
                 fields: Sequence[str],
                 max_block_size: int = 20,
                 window_size: int = 3,
                 step_size: int = 15,
                 sorting_key_fn: Optional[callable] = None,
                 *args,
                 **kwargs) -> None:
        """
           Split a block into smaller blocks using the sorted neighbourhood method.
        Args:
            window_size (int, optional): Window size. Defaults to 3.

        References:
            Sorted Neighborhood Methods,
            https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/folien/SS13/DPDC/DPDC_14_SNM.pdf
        """

        self.window_size = window_size
        self.step_size = step_size
        self.fields = fields
        self.max_block_size = max_block_size
        if sorting_key_fn is None:
            self.sorting_key_fn = self._default_sorting_key_fn
        else:
            self.sorting_key_fn = sorting_key_fn

    def _default_sorting_key_fn(self, x):
        return tuple(x[f] for f in self.fields)

    def __call__(self, block_id, block, *args, **kwargs):
        """
        Split a block into smaller blocks using the sorted neighbourhood method.
        Args:
            block (list): Block to be split.
        Returns:
            list: List of smaller blocks.
        """
        if len(block) <= self.max_block_size:
            yield block_id, block
        else:
            block = sorted(block, key=self.sorting_key_fn)

            for window in sliding_window(block, self.window_size, self.step_size):
                yield block_id, window


if __name__ == '__main__':
    import os
    from pathlib import Path
    import pandas as pd

    HERE = Path(os.path.dirname(__file__))
    DATA_DIR = HERE.parent.parent.parent / 'data'

    df = pd.read_csv(DATA_DIR / 'nvc50k.zip')
    df = df.fillna('')

    splitter = SortedNeighbourhoodBlockSplitter(fields=["surname", "dob"],
                                                window_size=3,
                                                step_size=1)

    for group_id, group in df.groupby("givenname"):
        for block_id, block in splitter(group_id, group.to_dict(orient="records")):
            print(pd.DataFrame(block))
            print("-" * 100)
