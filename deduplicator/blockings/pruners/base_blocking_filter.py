#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class BaseBlockingFilter:

    def __call__(self, *args, **kwargs):
        return False
