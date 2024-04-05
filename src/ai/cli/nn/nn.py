import importlib
import os

from click import Group, group


@group("nn", cls=Group, help="Commands for Neural Networks")
def main():
    pass


command_lines = sorted(
    [
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(os.path.dirname(__file__)))
        if f.endswith(".py") and not f.startswith("__init__") and not f == "nn.py"
    ]
)

[
    main.add_command(importlib.import_module(f"{__package__}.{c}").main, c)
    for c in command_lines
]
