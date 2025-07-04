from __future__ import annotations

import re
from typing import ClassVar, Optional, Self

from torch.utils.data import get_worker_info

from ..utils.func import classproperty
from .log import LoggableMixin


class Base(LoggableMixin):
    identification_name: ClassVar[str] = "type"

    @classmethod
    def all_subclasses(cls):
        return cls.__subclasses__() + [
            g for s in cls.__subclasses__() for g in s.all_subclasses()
        ]

    @classmethod
    def get_cls(cls, id: str) -> type[Self]:
        """Get the class corresponding to the identifier"""
        for sub in cls.all_subclasses():
            if id in sub.get_identifiers():
                return sub

        available_options = ", ".join(cls.get_subclass_identifiers())
        raise ValueError(
            f"Could not find a subclass with {cls.identification_name} '{id}'. "
            f"Available options are: [{available_options}]"
        )

    @classmethod
    def from_config(cls, config: dict | str | list) -> Self:
        """Create instance from config data (dict with type field, string identifier, or list for auto-conversion)"""
        if isinstance(config, str):
            return cls.from_identifier(config)
        elif isinstance(config, dict):
            if cls.identification_name in config:
                # Extract the type and create instance
                config_copy = config.copy()
                type_id = config_copy.pop(cls.identification_name)
                target_cls = cls.get_cls(type_id)
                return target_cls(**config_copy)
            else:
                # Try to create instance of this class directly
                return cls(**config)
        elif isinstance(config, list):
            # Special handling for classes that inherit from list[T]
            # Check if this class inherits from list
            has_list_base = any(
                hasattr(base, "__origin__") and base.__origin__ is list
                for base in getattr(cls, "__orig_bases__", [])
            )

            if has_list_base:
                # For list-inheriting classes, use the type conversion system
                from myconf.type_conversion import _format

                return _format(config, cls)
            else:
                # For regular classes, map list items to constructor arguments based on annotations
                annotations = getattr(cls, "__annotations__", {})
                if not annotations:
                    # If no annotations, pass list directly to constructor
                    return cls(*config)

                # Map list items to annotated fields in order
                annotated_fields = [
                    name
                    for name, annotation in annotations.items()
                    if not hasattr(annotation, "__origin__")
                    or annotation.__origin__ is not type(None)
                ]

                # Create kwargs from list mapping to annotated fields
                kwargs = {}
                for i, value in enumerate(config):
                    if i < len(annotated_fields):
                        kwargs[annotated_fields[i]] = value

                return cls(**kwargs)
        else:
            raise ValueError(
                f"Config must be string, dict, or list, got {type(config)}"
            )

    @classmethod
    def from_identifier(cls, identifier: str) -> Self:
        """Find and instantiate a subclass by identifier (case insensitive)"""
        identifier_lower = identifier.lower()

        for sub in cls.all_subclasses():
            # Check all possible identifiers for this subclass
            for sub_identifier in sub.get_identifiers():
                if sub_identifier.lower() == identifier_lower:
                    return sub()

        available_options = ", ".join(cls.get_subclass_identifiers())
        raise ValueError(
            f"Could not find a subclass with {cls.identification_name} '{identifier}'. "
            f"Available options are: [{available_options}]"
        )

    @classmethod
    def spaced_name(cls):
        """Get the class name with spaces between the words"""
        return re.sub(r"(?<![A-Z])([A-Z])", r" \1", cls.__name__).strip()

    @classmethod
    def get_identifiers(cls):
        """Get the identifiers that can be used to build the object"""
        name = cls.__name__
        spaced_name = cls.spaced_name()
        return {name, name.lower(), spaced_name, spaced_name.lower()}

    @classmethod
    def get_subclass_identifiers(cls):
        """Get the identifiers of the subclasses that can be used to build the object"""
        return set().union(*[sub.get_identifiers() for sub in cls.all_subclasses()])

    @classmethod
    def get_module_class_names(cls, module: type):
        """Get the names of the classes in the module that can be used for auto creating a class in our context"""
        return [name for name in dir(module) if not name.startswith("__")]


class BaseMP(Base):
    num_workers: Optional[int] = None

    @classproperty
    def is_mp(cls):
        return cls.worker_id is not None

    @classproperty
    def worker_id(cls):
        if worker_info := get_worker_info():
            return worker_info.id
        return None

    @classmethod
    def log_extras(cls):
        """Manages logging for a single instance or a worker instance"""
        txt = super().log_extras()
        if cls.is_mp:
            return txt + f" [Worker°{cls.worker_id}]"
        return txt
