#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# based on: https://github.com/microsoft/archai/blob/6cdefa40f6f91ac12198e648a05b3ea839fae7e5/archai/common/timing.py # noqa: E501
import gc
import logging
import os
import sys
import timeit
from functools import wraps
from typing import Dict, Optional, Callable

import psutil
from runstats import Statistics

_time_stats: Dict[str, Statistics] = {}
_ram_stats: Dict[str, Statistics] = {}
_max_ram_stats: Dict[str, Statistics] = {}


def add_timing(name: str, elapsed: float, no_print: bool = True) -> Statistics:
    global _time_stats

    stats = _time_stats.get(name, None)
    if stats is None:
        stats = Statistics()
        _time_stats[name] = stats
    stats.push(elapsed)

    if not no_print:
        logging.info('Timing "{}": {}s'.format(name, elapsed))
    return stats


def add_ram_usage(name: str, delta: float, max_ram: float, no_print: bool = True) \
        -> Statistics:
    global _ram_stats, _max_ram_stats

    stats = _ram_stats.get(name, None)
    max_stats = _max_ram_stats.get(name, None)
    if stats is None:
        stats, max_stats = Statistics(), Statistics()
        _ram_stats[name] = stats
        _max_ram_stats[name] = max_stats
    stats.push(delta)
    if sys.platform == 'darwin':
        max_stats.push(max_ram)
    else:
        max_stats.push(max_ram * (1 << 10))

    if not no_print:
        logging.info('RAMUsage "{}": {} GB'.format(name, delta / (1 << 30)))
        logging.info('MaxRAMUsage "{}": {} GB'.format(name, max_ram / (1 << 30)))
    return stats


def get_all_timings() -> Dict[str, Statistics]:
    global _time_stats
    return {f'TimeInSecondsPerCall@{k}': v.mean() for k, v in _time_stats.items()}


def get_all_ram_usages() -> Dict[str, Statistics]:
    global _ram_stats, _max_ram_stats
    stats = {
        f'RAMIncreaseInGBsAfterCall@{k}': float(v.mean()) / (1 << 30)
        for k, v in _ram_stats.items()
    }
    for k, v in _max_ram_stats.items():
        stats[f'RAMPeakInGBsAfterCall@{k}'] = float(v.mean()) / (1 << 30)
    return stats


def clear() -> None:
    global _time_stats, _ram_stats, _max_ram_stats
    _time_stats.clear()
    _ram_stats.clear()
    _max_ram_stats.clear()


def MeasureTime(f_py: Optional[Callable] = None,
                no_print: bool = True,
                disable_gc: bool = False,
                name: Optional[str] = None) -> Callable:
    assert callable(f_py) or f_py is None

    def _decorator(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
            gcold = gc.isenabled()
            if disable_gc:
                gc.disable()
            start_time = timeit.default_timer()
            try:
                result = f(*args, **kwargs)
            finally:
                elapsed = timeit.default_timer() - start_time
                if disable_gc and gcold:
                    gc.enable()
                fname = name or f.__name__
                add_timing(fname, elapsed, no_print=no_print)
            return result

        return _wrapper

    return _decorator(f_py) if callable(f_py) else _decorator


class MeasureBlockTime:
    def __init__(self, name: str, no_print: bool = True, disable_gc: bool = False):
        self.name = name
        self.no_print = no_print
        self.disable_gc = disable_gc

    def cur_elapsed(self) -> float:
        return timeit.default_timer() - self.start_time

    def __enter__(self):
        self.gcold = gc.isenabled()
        if self.disable_gc:
            gc.disable()
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, ty, val, tb):
        self.elapsed = self.cur_elapsed()
        if self.disable_gc and self.gcold:
            gc.enable()
        self.stats = add_timing(self.name, self.elapsed, no_print=self.no_print)
        return False  # re-raise any exceptions


class MeasureBlockRAM:
    def __init__(self, name: str, no_print: bool = True, disable_gc: bool = False):
        self.name = name
        self.no_print = no_print
        self.disable_gc = disable_gc
        self.pr = psutil.Process(os.getpid())

    def cur_ram_delta(self) -> float:
        return self.pr.memory_info().rss - self.start_ram

    def __enter__(self):
        self.gcold = gc.isenabled()
        if self.disable_gc:
            gc.disable()
        self.start_ram = self.pr.memory_info().rss
        return self

    def __exit__(self, ty, val, tb):
        self.ram_delta = self.cur_ram_delta()
        self.cur_max_ram = self.pr.memory_info().rss // 1024
        if self.disable_gc and self.gcold:
            gc.enable()
        self.stats = add_ram_usage(self.name, self.ram_delta, self.cur_max_ram,
                                   no_print=self.no_print)
        return False
