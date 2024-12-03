from __future__ import annotations

import re
from enum import IntEnum
from inspect import isclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pydantic import PrivateAttr
from torch.utils.data import get_worker_info

from ..registry import REGISTER
from ..utils.func import classproperty
from .log import Loggable

if TYPE_CHECKING:
    from ..launch import Main


class ActionEnum(IntEnum):
    fit = 0
    val = 1


class Base(Loggable):
    _root_config: Main | None = PrivateAttr(None)

    identification_name: ClassVar[str] = "type"

    def model_post_init(self, __context):
        self.info(f"Initialized {self.__class__.__name__}")

    @property
    def root_config(self) -> Main:
        return self._root_config

    def __init_subclass__(cls, buildable=True, **kwargs):
        super().__init_subclass__(**kwargs)
        # Manually check if the Model is buildable
        cls.is_buildable = buildable

    @classmethod
    def all_subclasses(cls):
        return cls.__subclasses__() + [
            g for s in cls.__subclasses__() for g in s.all_subclasses()
        ]

    @classmethod
    def from_config(
        cls, config: dict | str | Any | list, root: dict | str | Any = {}
    ) -> Self:
        """
        Function to be called whenever an object wants to use the identifier in the config to build its own object
        """
        if isinstance(config, cls):
            return config

        if isinstance(config, list):
            return [cls.from_config(c, root) for c in config]

        try:
            if isinstance(config, str):
                id = config
                config = {}
            else:
                id = config.pop(cls.identification_name)
        except KeyError:
            raise ValueError(
                f"Could not find key '{cls.identification_name}' in the config. "
                f"Please provide a '{cls.identification_name}' to build the object.\n"
                f"Current config:\n{config}"
            )

        sub = cls.get_cls(id)
        built = sub(**config)
        try:
            built.post_config(root)
        except:
            cls.error("Error while post-configuring", built, "\nwith config", root)
            raise
        return built

    def post_config(self, config: dict):
        """Called right after the object is built

        Args:
            config (dict): The root config used to build the object
        """
        ...

    @classmethod
    def get_cls(cls, id: str) -> type[Self]:
        for sub in cls.all_subclasses():
            if id in sub.get_identifiers():
                if sub.is_buildable:
                    return sub
                else:
                    raise ValueError(
                        f"You cannot directly build an object of type {sub.__name__}. "
                        f"Please use one of its subclasses."
                    )

        available_options = ", ".join(cls.get_subclass_identifiers())
        raise ValueError(
            f"Could not find a subclass with {cls.identification_name} '{id}'. "
            f"Available options are: [{available_options}]"
        )

    @classmethod
    def spaced_name(cls):
        return re.sub(r"(?<![A-Z])([A-Z])", r" \1", cls.__name__).strip()

    @classmethod
    def get_identifiers(cls):
        name = cls.__name__
        spaced_name = cls.spaced_name()
        return {name, name.lower(), spaced_name, spaced_name.lower()}

    @classmethod
    def get_subclass_identifiers(cls):
        return set().union(
            *[sub.get_identifiers() for sub in cls.__subclasses__() if sub.is_buildable]
        )

    @classmethod
    def from_config_model(cls, config: dict) -> Self:
        config = config.copy()
        source = config.pop("type")

        src_cls = REGISTER[source]

        if hasattr(src_cls, "config"):
            src_config = src_cls.config(**config)
        else:
            cls.info(
                f"{src_cls.__name__} has no config. Falling back to default constructor."
            )
            return src_cls(**config)

        return src_cls(src_config)

    @classmethod
    def create_classes(
        cls, namespace: dict[str, Any], module: type, selected_names: list[str] = None
    ):
        """Automatically create classes from a module in the specified namespace

        Args:
            namespace (dict[str, Any]): The namespace to create the classes in. Please use globals()
            module (type): The module to create the classes from
            selected_names (list[str], optional): The names of the classes to create. Defaults to None.
        """
        if selected_names is None:
            selected_names = [name for name in dir(module) if not name.startswith("__")]

        for cls_name in selected_names:
            try:
                module_cls = getattr(module, cls_name)
                if isclass(module_cls):
                    namespace[cls_name] = type(cls_name, (cls, module_cls), {})
                    # Update the class's module to match the caller's module
                    namespace[cls_name].__module__ = namespace["__name__"]
            except:
                cls.error(f"Error while creating class {cls_name} from module {module}")
                raise


class BaseMP(Base):
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
