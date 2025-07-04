"""Utility functions for MyConf"""

from typing import get_origin

from .constants import MYCONF_PROPERTIES_ATTR


def is_myconf_class(cls):
    """Check if a class is a MyConf class"""
    return hasattr(cls, MYCONF_PROPERTIES_ATTR)


def is_class_var(annotation):
    """Check if annotation is ClassVar"""
    if hasattr(annotation, "__origin__"):
        origin = get_origin(annotation)
        return origin and getattr(origin, "_name", None) == "ClassVar"
    # Also check string representations
    if isinstance(annotation, str) and annotation.startswith("ClassVar"):
        return True
    return False
