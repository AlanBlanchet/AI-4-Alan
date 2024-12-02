from __future__ import annotations

import io
import os
from functools import cache
from typing import Any, ClassVar

from pydantic import BaseModel
from rich.console import Console


class Color(BaseModel):
    red: ClassVar[str] = "\x1b[31m"
    green: ClassVar[str] = "\x1b[32m"
    yellow: ClassVar[str] = "\x1b[33m"
    blue: ClassVar[str] = "\x1b[34m"
    magenta: ClassVar[str] = "\x1b[35m"
    cyan: ClassVar[str] = "\x1b[36m"


class Loggable:
    log_name: ClassVar[str] = "main"
    color = None

    _logged_once = []

    @classmethod
    def _log_prefix(self) -> str:
        return f"[{self.log_name.capitalize()}] " if self.log_name else ""

    @classmethod
    def log(cls, *msg: list[Any], table=False):
        if os.getenv("MAIN_PROCESS", "1") == "1":
            if cls.color:
                print(
                    cls.color
                    + cls._log_prefix()
                    + "\x1b[0m"
                    + ("\n" if table else "")
                    + " ".join([str(m) for m in msg])
                )
            else:
                print(cls._log_prefix() + " ".join([str(m) for m in msg]))

    @classmethod
    @cache
    def log_once(cls, *msg: list[Any], table=False):
        cls.log(*msg, table=table)

    def log_table(self, table, width=100):
        console = Console(file=io.StringIO(), width=width)
        console.print(table)
        output = console.file.getvalue()
        self.log(output, table=True)
