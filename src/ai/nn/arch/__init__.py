import glob
import importlib
import os
from pathlib import Path

# Get the current directory
current_dir = Path(__file__).parent

# Import all modules
for module_p in current_dir.iterdir():
    init_p = module_p / "__init__.py"
    if init_p.exists():
        importlib.import_module(f".{module_p.name}", __package__)
