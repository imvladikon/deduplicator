#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib
import logging
import sys
from functools import lru_cache

POSTGRES_IMPORT_ERROR = "{0} requires the postgresql access library " \
                        "but it was not found in your environment. " \
                        "Checkout the instructions and follow the ones " \
                        "that match your environment: `pip install psycopg2`"

DUCKDB_IMPORT_ERROR = "{0} requires the duckdb client library " \
                      "but it was not found in your environment. " \
                      "Checkout the instructions and follow the ones " \
                      "that match your environment: `pip install duckdb`"


@lru_cache(maxsize=10)
def is_package_available(package_name: str) -> bool:
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None and package_spec.has_location


_psycopg2_is_found = is_package_available("psycopg2")
_duckdb_is_found = is_package_available("duckdb")

IMPORT_ERRORS_MAPPING = {
    "postgres": (_psycopg2_is_found, POSTGRES_IMPORT_ERROR),
    "duckdb": (_duckdb_is_found, DUCKDB_IMPORT_ERROR),
}


def check_backend_availability(obj, backend: str, on_error="raise") -> bool:
    """Check if the backend is available in the environment.
    alternative way is using https://pypi.org/project/necessary/, maybe in the future

    Args:
    obj: object to check the backend for
    backend: name of the backend to check
    on_error: what to do if the backend is not available, one of "raise", "warn", "ignore"
    """

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    to_check = IMPORT_ERRORS_MAPPING.get(backend)
    if to_check is not None:
        is_found, error_msg = to_check
        if not is_found:
            msg = error_msg.format(name)
            if on_error == "raise":
                raise ImportError(error_msg.format(name))
            elif on_error == "warn":
                import warnings

                warnings.warn(msg, UserWarning)
            else:
                logging.info(msg)
    else:
        is_found = is_package_available(backend)
        if not is_found:
            msg = f"Unknown backend: {backend}"
            if on_error == "raise":
                raise ImportError(msg)
            elif on_error == "warn":
                import warnings

                warnings.warn(msg, UserWarning)
            else:
                logging.info(msg)
    return is_found


def iter_subclasses(cls):
    for subcls in cls.__subclasses__():
        yield subcls
        yield from iter_subclasses(subcls)


def locate_class(module_name, class_name):
    module = sys.modules[module_name]
    try:
        klass = getattr(module, class_name)
    except:
        klass = None
    return klass
