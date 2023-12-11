#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import dataclasses
import inspect
import json
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentTypeError
from copy import copy
from enum import Enum
from functools import wraps
from inspect import isclass
from pathlib import Path
from typing import (Any,
                    Callable,
                    Dict,
                    Generic,
                    Iterable,
                    NewType,
                    Optional,
                    Tuple,
                    TypeVar,
                    Union,
                    get_type_hints)

PathLikeType = Union[str, Path]
DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)
T = TypeVar('T')


def string_to_bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    :param v:
    :return:
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(f"Truthy value expected: got {v} but expected one of "
                                f"yes/no, true/false, t/f, y/n, 1/0 (case insensitive).")


class ArgumentExtendedParser(Generic[T], argparse.ArgumentParser):
    """
    Extension of `argparse.ArgumentParser`
    based on:
    1. HFArgumentParser
    https://github.com/huggingface/transformers/blob/17a55534f5e5df10ac4804d4270bf6b8cc24998d/src/transformers/hf_argparser.py#L119
    2. https://github.com/eladrich/pyrallis

    Optional: To create sub argument groups use the `_argument_group_name`
    attribute in the dataclass.

    Example:
        ```python
        @dataclass
        class BenchmarkArguments:
             _argument_group_name = "benchmarking"

            num_threads: Optional[int] = field(
                default=None,
                metadata={
                    "help": "Number of the threads",
                    "shortcut": "-n"
                },
            )

        @fire()
        def main(args: BenchmarkArguments):
            print(args.num_threads)
        ```
    """
    dataclass_types: Iterable[DataClassType]

    def __init__(
            self, dataclass_types: Union[DataClassType, Iterable[DataClassType]], **kwargs
    ):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types
                for which we will "fill" instances with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
        """
        # To make the default appear when using --help
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = ArgumentDefaultsHelpFormatter
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = list(dataclass_types)
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    # flake8: noqa: C901
    @staticmethod
    def _parse_dataclass_field(parser: argparse.ArgumentParser, field: dataclasses.Field):
        field_name = f"--{field.name}"
        kwargs = field.metadata.copy()
        # field.metadata is not used at all by Data Classes,
        # it is provided as a third-party extension mechanism.
        if isinstance(field.type, str):
            raise RuntimeError(
                "Unresolved type detected, which should have been done with the help of "
                "`typing.get_type_hints` method by default"
            )

        origin_type = getattr(field.type, "__origin__", field.type)
        if origin_type is Union:
            if str not in field.type.__args__ and (
                    len(field.type.__args__) != 2 or type(None) not in field.type.__args__
            ):
                raise ValueError("Only `Union[X, NoneType]` (i.e., `Optional[X]`) "
                                 "is allowed for `Union` because the argument parser "
                                 "only supports one type per argument. "
                                 f"Problem encountered in field '{field.name}'.")
            if type(None) not in field.type.__args__:
                # filter `str` in Union
                field.type = (
                    field.type.__args__[0]
                    if field.type.__args__[1] == str
                    else field.type.__args__[1]
                )
                origin_type = getattr(field.type, "__origin__", field.type)
            elif bool not in field.type.__args__:
                # filter `NoneType` in Union (except for `Union[bool, NoneType]`)
                field.type = (
                    field.type.__args__[0]
                    if isinstance(None, field.type.__args__[1])
                    else field.type.__args__[1]
                )
                origin_type = getattr(field.type, "__origin__", field.type)

        # A variable to store kwargs for a boolean field, if needed
        # so that we can init a `no_*` complement argument (see below)
        bool_kwargs = {}
        if isinstance(field.type, type) and issubclass(field.type, Enum):
            kwargs["choices"] = [x.value for x in field.type]
            kwargs["type"] = type(kwargs["choices"][0])
            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            else:
                kwargs["required"] = True
        elif field.type is bool or field.type == Optional[bool]:
            # Copy the currect kwargs
            # to use to instantiate a `no_*` complement argument below.
            # We do not initialize it here because the `no_*` alternative
            # must be instantiated after the real argument
            bool_kwargs = copy(kwargs)

            # Hack because type=bool in argparse does not behave as we want.
            kwargs["type"] = string_to_bool
            if field.type is bool or (
                    field.default is not None and field.default is not dataclasses.MISSING
            ):
                # Default value is False if we have no default when of type bool.
                default = False if field.default is dataclasses.MISSING else field.default
                # This is the value that will get picked
                # if we don't include --field_name in any way
                kwargs["default"] = default
                # This tells argparse we accept 0 or 1 value after --field_name
                kwargs["nargs"] = "?"
                # This is the value that will get picked
                # if we do --field_name (without value)
                kwargs["const"] = True
        elif isclass(origin_type) and issubclass(origin_type, list):
            kwargs["type"] = field.type.__args__[0]
            kwargs["nargs"] = "+"
            if field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            elif field.default is dataclasses.MISSING:
                kwargs["required"] = True
        else:
            kwargs["type"] = field.type
            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            else:
                kwargs["required"] = True
        field_names = [field_name]
        shortcut = kwargs.pop("shortcut", None)
        if shortcut is not None:
            field_names.append(shortcut)
        if "_" in field_name:
            field_names.append(field_name.replace("_", "-"))
        parser.add_argument(*field_names, **kwargs)

        # Add a complement `no_*` argument for a boolean field AFTER the initial field
        # has already been added.
        # Order is important for arguments with the same destination!
        # We use a copy of earlier kwargs because
        # the original kwargs have changed a lot before reaching down
        # here and we do not need those changes/additional keys.
        if field.default is True and (field.type is bool or field.type == Optional[bool]):
            bool_kwargs["default"] = False
            field_names = [f"--no_{field.name}"]
            if "_" in field.name:
                field_names.append(f"--no-{field.name.replace('_', '-')}")
            parser.add_argument(
                *field_names, action="store_false", dest=field.name, **bool_kwargs
            )

    def _add_dataclass_arguments(self, dtype: DataClassType):
        if hasattr(dtype, "_argument_group_name"):
            parser = self.add_argument_group(dtype._argument_group_name)
        else:
            parser = self

        try:
            type_hints: Dict[str, type] = get_type_hints(dtype)
        except NameError:
            raise RuntimeError(f"Type resolution failed for f{dtype}. "
                               f"Try declaring the class in global scope or removing "
                               "line of `from __future__ import annotations` which opts "
                               "in Postponed Evaluation of Annotations (PEP 563)")

        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            # check if the type is a dataclass
            if dataclasses.is_dataclass(field.type):
                self._add_dataclass_arguments(field.type)
            else:
                field.type = type_hints[field.name]
                self._parse_dataclass_field(parser, field)

    def parse_args(self,
                   args=None,
                   return_remaining_strings=False,
                   look_for_args_file=True,
                   args_filename=None) -> Union[DataClass, Tuple[DataClass, ...]]:
        """
        Parse command-line args into instances of the specified dataclass types.
        This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args
        Args:
            args:
                List of strings to parse. The default is taken from sys.argv.
                (same as argparse.ArgumentParser)
            return_remaining_strings:
                If true, also return a list of remaining argument strings.
            look_for_args_file:
                If true, will look for a ".args" file with the same base name
                as the entry point script for this
                process, and will append its potential content to the command line args.
            args_filename:
                If not None, will uses
                this file instead of the ".args" file specified in the previous argument.
        Returns:
            Tuple consisting of:
                - the dataclass instances
                in the same order as they were passed to the initializer.abspath
                - if applicable, an additional namespace for more
                (non-dataclass backed) arguments added to the parser after initialization.
                - The potential list of remaining argument strings.
                (same as argparse.ArgumentParser.parse_known_args)
        """
        if args_filename or (look_for_args_file and len(sys.argv)):
            if args_filename:
                args_file = Path(args_filename)
            else:
                args_file = Path(sys.argv[0]).with_suffix(".args")

            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + args if args is not None else fargs + sys.argv[1:]
                # in case of duplicate arguments the first one has precedence
                # so we append rather than prepend.
        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        arguments = dict(vars(namespace))

        def _init_obj(dataclass_types):
            for dtype in dataclass_types:
                keys = {f.name for f in dataclasses.fields(dtype) if f.init}
                inputs = {k: v for k, v in arguments.items() if k in keys}
                for k in keys:
                    if hasattr(namespace, k):
                        delattr(namespace, k)
                obj = dtype(**inputs)
                return obj

        for dtype in self.dataclass_types:
            for field in dataclasses.fields(dtype):
                if dataclasses.is_dataclass(field.type):
                    obj = _init_obj([field.type])
                    arguments[field.name] = obj

        obj = _init_obj(self.dataclass_types)
        outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(f"Some specified arguments are not used by "
                                 f"the ArgumentParser: {remaining_args}")
            if len(outputs) == 1:
                return outputs[0]
            else:
                return (*outputs,)

    def parse_dict(
            self, args: Dict[str, Any], allow_extra_keys: bool = False
    ) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all,
        instead uses a dict and populating the dataclass types.
        Args:
            args (`dict`):
                dict containing config values
            allow_extra_keys (`bool`, *optional*, defaults to `False`):
                Defaults to False. If False, will raise an exception
                if the dict contains keys that are not parsed.
        Returns:
            Tuple consisting of:
                - the dataclass instances
                in the same order as they were passed to the initializer.
        """
        unused_keys = set(args.keys())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in args.items() if k in keys}
            unused_keys.difference_update(inputs.keys())
            obj = dtype(**inputs)
            outputs.append(obj)
        if not allow_extra_keys and unused_keys:
            raise ValueError(
                f"Some keys are not used by the HfArgumentParser: {sorted(unused_keys)}"
            )
        return tuple(outputs)

    def parse_file(
            self, filepath: PathLikeType, allow_extra_keys: bool = False
    ) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all,
        instead loading a file and populating the dataclass types.
        Args:
            filepath (`str` or `os.PathLike`):
                File name of the json file to parse
            allow_extra_keys (`bool`, *optional*, defaults to `False`):
                Defaults to False. If False, will raise an exception
                if the json file contains keys that are not parsed.
        Returns:
            Tuple consisting of:
                - the dataclass instances
                in the same order as they were passed to the initializer.
        """
        filepath = Path(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.suffix == 'json':
                data = json.load(f)
            elif filepath.suffix == 'yaml':
                import yaml

                data = yaml.safe_load(f.read())
            else:
                raise ValueError(f"Unrecognized file extension {filepath.suffix}")
        outputs = self.parse_dict(data, allow_extra_keys=allow_extra_keys)
        return tuple(outputs)


def fire() -> Callable:
    """
    Helper function to fire the ArgumentExtendedParser.
    e.g.
    ```python
    @fire()
    def main(args: DataClass):
        print(args)
    ```
    :return:
    """

    def wrapper_outer(fn):
        @wraps(fn)
        def wrapper_inner(*args, **kwargs):
            argspec = inspect.getfullargspec(fn)
            argtypes = [argspec.annotations[a] for a in argspec.args]
            parser = ArgumentExtendedParser(argtypes)
            cfg = parser.parse_args()
            if isinstance(cfg, tuple):
                return fn(*cfg, *args, **kwargs)
            else:
                return fn(cfg, *args, **kwargs)

        return wrapper_inner

    return wrapper_outer
