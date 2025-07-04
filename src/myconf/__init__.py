"""
MyConf provides automatic type conversion, property management, and configuration handling.
"""

from .base import MyConf
from .core import Cast, Consumed, F
from .decorators import auto_convert
from .ide_signature import debug_ide_type_info
from .stub_generation import generate_stub_file

__all__ = [
    "MyConf",
    "F",
    "Cast",
    "Consumed",
    "auto_convert",
    "debug_ide_type_info",
    "generate_stub_file",
]

# Version info
__version__ = "1.0.0"
