import importlib
import os
from pathlib import Path

from click import Group, group


@group("ai", cls=Group)
def main():
    pass


command_lines = sorted(
    [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(os.path.dirname(__file__)))
        if f.endswith(".py") and not f.startswith("__init__") and not f == "ai.py"
    ]
)

groups = [
    f for f in Path(__file__).parent.iterdir() if f.is_dir() and f.name != "__pycache__"
]

[
    main.add_command(importlib.import_module(f"{__package__}.{c}").main, c)
    for c in command_lines
]

for g in groups:
    [
        main.add_command(
            importlib.import_module(f"{__package__}.{g.name}.{g.stem}").main
        )
        for f in g.iterdir()
        if f.suffix == ".py" and f.stem != "__init__"
    ]
