#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import deque
from itertools import islice, tee, combinations, chain
from operator import itemgetter

from scipy.special import comb as size_of_combinations
from typing import Any, Iterable, Iterator, Tuple, Sequence

from tqdm import tqdm


def batched(iterable: Iterable[Any], batch_size: int) -> Iterator[Tuple[Any]]:
    """
    Batch data into iterables/lists of length *n*.
    The last batch may be shorter.
    example: list(batched('ABCDEFG', 3))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]
    This recipe is from the ``itertools`` docs.
    This library also provides
    :func:`chunked`,
    which has a different implementation.
    """
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, batch_size))
        if not batch:
            break
        yield batch


def iter_pairwise(iterable):
    """
    Iterate over consecutive pairs:
        s -> (s[0], s[1]), (s[1], s[2]), (s[2], s[3]), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def all_pairs_for(seq: Sequence,
                  verbose: bool = False,
                  return_one_pair: bool = False) -> Iterator[Tuple[Any, Any]]:
    """
    iterate over all pairs of elements in a sequence (list, tuple, etc.)
    Args:
        seq: sequence of elements
        verbose: if True, show progress bar
        return_one_pair: if True, if the sequence has only one element,
        return the pair (element, element)
    Returns:
        iterator over all pairs of elements in the sequence
    """
    if return_one_pair and len(seq) == 1:
        value = next(iter(seq))
        yield value, value
        return

    it = combinations(seq, 2)
    if verbose:
        total = size_of_combinations(len(seq), 2)
        yield from tqdm(it, total=total)
    else:
        yield from it


def all_pairs_for_range(k: int, verbose: bool = False) -> Iterator[Tuple[int, int]]:
    """
    the same as all_pairs_for, but for a range of integers
    """
    it = combinations(range(k), 2)
    if verbose:
        total = size_of_combinations(k, 2)
        yield from tqdm(it, total=total)
    else:
        yield from it


def unique_everseen(iterable, key=None):
    """
    Yield unique elements, preserving order.

        >>> list(unique_everseen('AAAABBBCCDAABBB'))
        ['A', 'B', 'C', 'D']
        >>> list(unique_everseen('ABBCcAD', str.lower))
        ['A', 'B', 'C', 'D']

    Sequences with a mix of hashable and unhashable items can be used.
    The function will be slower (i.e., `O(n^2)`) for unhashable items.

    Remember that ``list`` objects are unhashable - you can use the *key*
    parameter to transform the list to a tuple (which is hashable) to
    avoid a slowdown.

        >>> iterable = ([1, 2], [2, 3], [1, 2])
        >>> list(unique_everseen(iterable))  # Slow
        [[1, 2], [2, 3]]
        >>> list(unique_everseen(iterable, key=tuple))  # Faster
        [[1, 2], [2, 3]]

    Similary, you may want to convert unhashable ``set`` objects with
    ``key=frozenset``. For ``dict`` objects,
    ``key=lambda x: frozenset(x.items())`` can be used.

    """
    seenset = set()
    seenset_add = seenset.add
    seenlist = []
    seenlist_add = seenlist.append
    use_key = key is not None

    for element in iterable:
        k = key(element) if use_key else element
        try:
            if k not in seenset:
                seenset_add(k)
                yield element
        except TypeError:
            if k not in seenlist:
                seenlist_add(k)
                yield element


def distinct_combinations(iterable, r):
    """Yield the distinct combinations of *r* items taken from *iterable*.

        >>> list(distinct_combinations([0, 0, 1], 2))
        [(0, 0), (0, 1)]

    Equivalent to ``set(combinations(iterable))``, except duplicates are not
    generated and thrown away. For larger input sequences this is much more
    efficient.

    """
    if r < 0:
        raise ValueError('r must be non-negative')
    elif r == 0:
        yield ()
        return
    pool = tuple(iterable)
    generators = [unique_everseen(enumerate(pool), key=itemgetter(1))]
    current_combo = [None] * r
    level = 0
    while generators:
        try:
            cur_idx, p = next(generators[-1])
        except StopIteration:
            generators.pop()
            level -= 1
            continue
        current_combo[level] = p
        if level + 1 == r:
            yield tuple(current_combo)
        else:
            generators.append(
                unique_everseen(
                    enumerate(pool[cur_idx + 1:], cur_idx + 1),
                    key=itemgetter(1),
                )
            )
            level += 1


def sliding_window(seq, window_size, step_size=1):
    """Return a sliding window of width *n* over *iterable*. with a step of k

        >>> list(sliding_window(range(6), 4))
        [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]

    If *iterable* has fewer than *n* items, then nothing is yielded:

        >>> list(sliding_window(range(3), 4))
        []

    For a variant with more features, see :func:`windowed`.
    """
    start, end = 0, window_size
    while True:
        window = deque(islice(seq, start, end))
        yield tuple(window)
        start += step_size
        end += step_size
        if len(window) < window_size:
            break


def flatten(iterable):
    return chain.from_iterable(iterable)


def take_unique_everseen(iterable, limit, key=None):
    """
    Yield each element while keeping track of the seen elements.
    """
    seenset = set()
    seenset_add = seenset.add
    seenlist = []
    seenlist_add = seenlist.append
    use_key = key is not None

    for element in iterable:
        k = key(element) if use_key else element
        try:
            if k not in seenset:
                seenset_add(k)
            if len(seenset) <= limit:
                yield element
        except TypeError:
            if k not in seenset:
                seenlist_add(k)
            if len(seenset) <= limit:
                yield element
