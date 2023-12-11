#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import logging
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm as tqdm_auto


class TqdmToLogger(io.StringIO):
    """
        Output stream for tqdm which will output to logger module instead of
        the stderr.
        Adapted from https://github.com/tqdm/tqdm/issues/313#issuecomment-267959111
        or https://stackoverflow.com/a/41224909
    """

    def __init__(self, logger: logging.Logger):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = logger.level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level
        self.fileno = lambda: False

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to.
        pass


def get_logger(name: str, logs_file_path: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)

    # FileHandler to log to files
    if logs_file_path is not None:
        Path(logs_file_path).parent.mkdir(parents=True, exist_ok=True)

        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)

        file_handler = logging.FileHandler(logs_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def progress_bar(iterable, total=None, desc=None, logger=None):
    """
        Wrapper around tqdm progress bar to redirect output to logger.
        Prints a progress bar update each 5 seconds on a new line.
    """
    if logger is None:
        logger = get_logger(__name__)

    tqdm_out = TqdmToLogger(logger)

    return tqdm_auto(iterable, total=total, mininterval=5, desc=desc, file=tqdm_out)


tqdm = progress_bar
