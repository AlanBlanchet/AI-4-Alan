from __future__ import annotations

import io
from _collections_abc import dict_items, dict_keys, dict_values
from functools import cache
from logging import FileHandler, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path
from pprint import pformat
from typing import Any, ClassVar, Literal

from devtools import pformat as pydantic_pformat
from pydantic import BaseModel, PrivateAttr
from rich.console import Console

from ..utils.env import AIEnv
from ..utils.func import classproperty


class Color(BaseModel):
    black: ClassVar[str] = "\x1b[30m"
    red: ClassVar[str] = "\x1b[31m"
    green: ClassVar[str] = "\x1b[32m"
    yellow: ClassVar[str] = "\x1b[33m"
    blue: ClassVar[str] = "\x1b[34m"
    magenta: ClassVar[str] = "\x1b[35m"
    cyan: ClassVar[str] = "\x1b[36m"
    orange: ClassVar[str] = "\x1b[91m"
    lime: ClassVar[str] = "\x1b[92m"
    gold: ClassVar[str] = "\x1b[93m"
    sky: ClassVar[str] = "\x1b[94m"
    pink: ClassVar[str] = "\x1b[95m"
    teal: ClassVar[str] = "\x1b[96m"
    white: ClassVar[str] = "\x1b[97m"


LOG = Path("log")


class Loggable(BaseModel):
    model_config = {"ignored_types": (classproperty,)}

    log_name: ClassVar[str] = "main"
    """The name to use when calling the log function"""
    color: ClassVar[str] = Color.white
    """Chose a color to use when printing"""

    _logged_once = []
    _logger: ClassVar[Logger] = PrivateAttr(None)
    _loggers: ClassVar[set[Logger]] = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.log_name not in cls._loggers:
            cls._loggers.add(cls.log_name)
            cls._logger = cls._configure_new_logger()

    @classmethod
    def _configure_new_logger(cls):
        """Configure subclass-specific log files."""
        logger = getLogger(f"AI-{cls.log_name}")

        if logger.hasHandlers():
            # Avoid adding duplicate handlers
            return logger

        log_p = AIEnv.tmp_log_p / cls.log_name
        log_p.mkdir(exist_ok=True)

        # FILES ====================
        # Set up handlers for debug, info, warn, and error logs
        file_fmt = Formatter(
            "PID %(process)d|T %(thread)d|%(asctime)s|%(levelname)s|%(message)s"
        )
        debug_handler = FileHandler(log_p / "debug.txt", mode="a")
        debug_handler.setFormatter(file_fmt)
        debug_handler.setLevel("DEBUG")

        info_handler = FileHandler(log_p / "info.txt", mode="a")
        info_handler.setFormatter(file_fmt)
        info_handler.setLevel("INFO")

        warn_handler = FileHandler(log_p / "warn.txt", mode="a")
        warn_handler.setFormatter(file_fmt)
        warn_handler.setLevel("WARNING")

        error_handler = FileHandler(log_p / "err.txt", mode="a")
        error_handler.setFormatter(file_fmt)
        error_handler.setLevel("ERROR")

        # Add handlers to the logger
        logger.addHandler(debug_handler)
        logger.addHandler(info_handler)
        logger.addHandler(warn_handler)
        logger.addHandler(error_handler)

        # CLI ====================

        # Optionally, log to console for info messages
        info_stream_handler = StreamHandler()
        info_stream_handler.setFormatter(
            Formatter(f"{cls.color}{cls._log_prefix()}\033[0m%(message)s")
        )
        info_stream_handler.setLevel("INFO")
        info_stream_handler.addFilter(lambda record: record.levelno == 20)  # INFO level
        logger.addHandler(info_stream_handler)

        # Add a specific handler for warnings to show messages in yellow
        warning_stream_handler = StreamHandler()
        warning_stream_handler.setFormatter(
            Formatter(
                f"{cls.color}{cls._log_prefix()}\033[0m{Color.yellow}%(message)s\033[0m"
            )
        )
        warning_stream_handler.setLevel("WARNING")
        warning_stream_handler.addFilter(
            lambda record: record.levelno == 30
        )  # WARNING level
        logger.addHandler(warning_stream_handler)

        # Add a specific handler for errors to show messages in red
        error_stream_handler = StreamHandler()
        error_stream_handler.setFormatter(
            Formatter(
                f"{cls.color}{cls._log_prefix()}\033[0m{Color.red}%(message)s\033[0m"
            )
        )
        error_stream_handler.setLevel("ERROR")
        error_stream_handler.addFilter(
            lambda record: record.levelno == 40
        )  # ERROR level
        logger.addHandler(error_stream_handler)

        # Set logger level to DEBUG to capture all logs
        logger.setLevel("DEBUG")

        return logger

    @classmethod
    def _log_prefix(cls) -> str:
        return f"[{cls.log_name.capitalize()}] " if cls.log_name else ""

    @classmethod
    def _resolve_msg(cls, msg: list[Any]):
        new_msg = []
        for m in [cls.log_extras(), *msg]:
            if isinstance(m, (dict_items, dict_keys, dict_values)):
                m = list(m)
            if isinstance(m, (dict, list, set)):
                new_msg.append(pformat(m))
            elif isinstance(m, BaseModel):
                new_msg.append(pydantic_pformat(m))
            else:
                new_msg.append(f"{m} ")
        return "".join(new_msg).strip()

    @classmethod
    def log_extras(cls) -> str:
        return ""

    @classmethod
    def debug(cls, *msg: list[Any]):
        cls._logger.debug(cls._resolve_msg(msg))

    @classmethod
    def info(cls, *msg: list[Any]):
        cls._logger.info(cls._resolve_msg(msg))

    @classmethod
    def warn(cls, *msg: list[Any]):
        cls._logger.warning(cls._resolve_msg(msg))

    @classmethod
    def error(cls, *msg: list[Any]):
        cls._logger.error(cls._resolve_msg(msg))

    @classmethod
    def log(cls, *msg: list[Any], type: Literal["err", "warn", "info"] = "info"):
        if type == "err":
            cls.error(*msg)
        elif type == "warn":
            cls.warn(*msg)
        cls.info(*msg)

    @classmethod
    @cache
    def log_once(cls, *msg: list[Any]):
        cls.info(*msg)

    @classmethod
    @cache
    def warn_once(cls, *msg: list[Any]):
        cls.warn(*msg)

    @classmethod
    def log_table(cls, table, width=100):
        console = Console(file=io.StringIO(), width=width)
        console.print(table)
        output = console.file.getvalue()
        cls.info(output, table=True)

    @classmethod
    def watch(cls, *args, **fn_kwargs):
        """Decorator that watches a function for errors and logs them."""

        def decorator(func, *args, **kwargs):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    cls.error(str(e))
                    raise e

            return wrapper

        return decorator(*args, **fn_kwargs)


# Define the main logger on loggable
Loggable._logger = Loggable._configure_new_logger()
