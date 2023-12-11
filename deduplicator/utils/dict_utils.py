#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools
from typing import Any, Dict


def nset_attr(obj: Any, attr: str, val: Any) -> None:
    """
    Set attribute of nested object.
    :param obj:
    :param attr:
    :param val:
    :return:
    """
    pre, _, post = attr.rpartition('.')
    return setattr(nget_attr(obj, pre) if pre else obj, post, val)


def nget_attr(obj: Any, attr: str, *args: Any) -> Any:
    """
    Get attribute of nested object.
    :param obj:
    :param attr:
    :param args:
    :return:
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def nhas_attr(obj: Any, attr: str) -> bool:
    """
    Check if attribute of nested object exists.
    :param obj:
    :param attr:
    :return:
    """
    if hasattr(obj, attr):
        return True
    try:
        nget_attr(obj, attr)
    except AttributeError:
        return False
    else:
        return True


def nupdateattrs(obj: Any, properties: Dict) -> None:
    """
    Update attributes of nested object.
    :param obj:
    :param attr:
    :param val:
    :return:
    """
    for k, v in properties.items():
        if nhas_attr(obj, k):
            nset_attr(obj, k, v)


def nget_item(d: Dict, key: str, default: Any = None) -> Any:
    item = d
    for key in key.split('.'):
        if key in item:
            item = item[key]
        else:
            return default
    return item


def nset_item(data: Dict, key: str, value: Any) -> None:
    keys = key.split('.')
    item = data
    for key in keys[:-1]:
        if key not in item:
            item[key] = {}
        item = item[key]
    item[keys[-1]] = value
