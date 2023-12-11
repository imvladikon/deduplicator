#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import wraps

from deduplicator.utils import check_backend_availability


class LineTimeProfiler:

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(LineTimeProfiler, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if check_backend_availability(self, "line_profiler", on_error="ignore"):
            global line_profiler
            import line_profiler

            self.line_profiler = line_profiler.LineProfiler()
        else:
            self.line_profiler = None

    def __call__(self, func):
        if self.line_profiler is None:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            self.line_profiler.add_function(func)
            self.line_profiler.enable_by_count()
            ret = func(*args, **kwargs)
            self.line_profiler.disable_by_count()
            self.line_profiler.print_stats()
            return ret

        return wrapper


class LineMemoryProfiler:

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(LineMemoryProfiler, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if check_backend_availability(self, "memory_profiler", on_error="ignore"):
            import memory_profiler

            self.line_profiler = memory_profiler.profile
        else:
            self.line_profiler = None

    def __call__(self, func):
        if self.line_profiler is None:
            return func

        return self.line_profiler(func)


global_line_time_profiler = LineTimeProfiler()
global_line_memory_profiler = LineMemoryProfiler()
