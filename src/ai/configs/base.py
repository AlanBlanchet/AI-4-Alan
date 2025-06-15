from __future__ import annotations

import inspect
import re
from copy import deepcopy
from enum import IntEnum
from inspect import isclass, signature
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Self

from pydantic import BaseModel, PrivateAttr, field_validator, model_validator
from pydantic._internal._decorators import DecoratorInfos
from pydantic._internal._model_construction import ModelMetaclass
from pydantic.fields import FieldInfo
from torch.utils.data import get_worker_info

from ..utils.func import classproperty
from .log import Loggable

if TYPE_CHECKING:
    from ..launch import Main


class ActionEnum(IntEnum):
    fit = 0
    val = 1


# class PydanticAutoMetaclass(ModelMetaclass):
#     def __new__(
#         mcs,
#         cls_name: str,
#         bases: tuple[type[Any], ...],
#         namespace: dict[str, Any],
#         **kwargs: Any,
#     ):
#         cls = super().__new__(mcs, cls_name, bases, namespace, **kwargs)

#         for name, field in cls.model_fields.items():
#             field: FieldInfo
#             mcls = field.annotation
#             if isclass(mcls) and issubclass(mcls, Base):
#                 mcls: Base
#                 validator_name = f"validate_{name}"
#                 if not hasattr(cls, validator_name):

#                     @validator(name)
#                     def auto_validator(cls, v, values):
#                         if v is None:
#                             v = {}
#                         return cls.from_config(v, values)

#                     setattr(cls, validator_name, auto_validator)

#         return cls


class Base(Loggable, metaclass=ModelMetaclass):
    _root_config: Main | None = PrivateAttr(None)

    auto_build: ClassVar[bool] = False
    """Allows to build the object even if no config is provided"""

    identification_name: ClassVar[str] = "type"

    default: ClassVar[str] = None
    """The default sub class to use if no type is provided"""

    def model_post_init(self, __context):
        self.info(f"Initialized {self.__class__.__name__}")

    @model_validator(mode="before")
    def base_model_validator(cls, config):
        return cls.configure(config)

    @classmethod
    def configure(cls, config: dict):
        """Configure the objects from the config

        Usefull when objects have relations between them. You can manually build the objects if you want
        or return the corresponding config.

        WARNING: Always call super.configure() to automatically parse the non configured objects
        """
        for k, v in cls.model_fields.items():
            try:
                if k in config:
                    mcls = v.annotation
                    if isinstance(mcls, type) and issubclass(mcls, Base):
                        if isinstance(config[k], mcls):
                            continue  # Object already built
                        elif mcls.identification_name in config[k]:
                            # There is a specific class to build
                            config[k] = mcls.from_config(config[k])
                        elif mcls.default is not None:
                            # There is a default class to build
                            config[k] = mcls.from_config(
                                {mcls.identification_name: mcls.default} | config[k]
                            )
                        else:
                            # We can try building the object directly manually
                            config[k] = mcls(**config[k])
            except Exception as e:
                raise Exception(f"Could not configure {k} in {cls.__name__}") from e
        return config

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
    def __get_pydantic_core_schema__(cls, _source_type: type[Base], _handler: Any):
        schema = cls.__dict__.get("__pydantic_core_schema__")
        if schema is None:
            remove_decorators = set()
            for name, field in _source_type.model_fields.items():
                field: FieldInfo
                mcls = field.annotation
                if isclass(mcls) and issubclass(mcls, Base):
                    mcls: Base

                    if mcls.auto_build:
                        # Allow None to validate by default with the validator
                        field.validate_default = True
                        field.default = None

                    # The futur generated validator name
                    validator_default_name = f"validate_{name}_default"

                    if not hasattr(_source_type, validator_default_name):
                        # Get the potential user validator
                        validator_name = f"validate_{name}"
                        user_validator_name = None
                        if hasattr(_source_type, validator_name):
                            remove_decorators.add(validator_name)
                            user_validator_name = validator_name

                        # Create one
                        setattr(
                            _source_type,
                            validator_default_name,
                            cls._create_decorator(name, mcls, user_validator_name),
                        )

            # Remove user validators from pydantic since we now have our own
            for name in remove_decorators:
                _source_type.__pydantic_decorators__.field_validators.pop(name)

            decorators = deepcopy(_source_type.__pydantic_decorators__)
            new_decorators = DecoratorInfos.build(cls)
            _source_type.__pydantic_decorators__ = decorators
            decorators.field_validators.update(new_decorators.field_validators)

        return super().__get_pydantic_core_schema__(_source_type, _handler)

    @staticmethod
    def _create_decorator(name, mcls, user_validator_name):
        @field_validator(name, mode="before")
        def auto_validator(
            cls,
            value,
            values,
            mcls=mcls,
            user_validator_name: str = user_validator_name,
        ):
            """Automatically build the object from the config"""
            try:
                if hasattr(values, "data") and isinstance(values.data, dict):
                    values = values.data

                if value is None:
                    value = {}
                elif isinstance(value, mcls):
                    return value

                # Use the user validator if exists
                if user_validator_name is not None:
                    user_validator = getattr(cls, user_validator_name)
                    n = len(signature(user_validator).parameters)

                    if n == 1:
                        value = user_validator(value)
                    elif n == 2:
                        value = user_validator(value, values)

                    # The validator has already built the object
                    if isinstance(value, mcls):
                        return value

                if mcls.identification_name not in value and mcls.default is not None:
                    # Add the default class to build if no type is provided
                    value[mcls.identification_name] = mcls.default

                # Build the object
                if mcls.identification_name in value:
                    obj = mcls.from_config(value)
                else:
                    obj = mcls(**value)

                return obj
            except Exception as e:
                cls.error(
                    f"Error while building {mcls.__name__}.", f"value is \n{value}\n", e
                )
                raise Exception(
                    f"Error while building {mcls.__name__} from {cls.__name__}"
                ) from e

        return auto_validator

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

        if "_args" in config and inspect.isclass(sub):
            args = config.pop("_args")
            if issubclass(sub, BaseModel):
                params = list(sub.model_fields)
            else:
                signature = inspect.signature(sub.__init__)
                params = list(signature.parameters)[1:]  # remove self

            kwargs = {name: arg for name, arg in zip(params, args)}
            config.update(kwargs)

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
        """Get the class corresponding to the identifier"""
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
        return set().union(
            *[sub.get_identifiers() for sub in cls.__subclasses__() if sub.is_buildable]
        )

    @classmethod
    def get_module_class_names(cls, module: type):
        """Get the names of the classes in the module that can be used for auto creating a class in our context"""
        return [name for name in dir(module) if not name.startswith("__")]

    @classmethod
    def create_classes(
        cls,
        *,
        namespace: dict[str, Any],
        module: type,
        selected_names: list[str] = None,
        required_base: type = None,
    ):
        """Automatically create classes from a module in the specified namespace

        Args:
            namespace (dict[str, Any]): The namespace to create the classes in. Please use globals()
            module (type): The module to create the classes from
            selected_names (list[str], optional): The names of the classes to create. Defaults to None.
            required_base (type, optional): The base class that the created classes should inherit from. Defaults to None.
        """
        if selected_names is None:
            selected_names = cls.get_module_class_names(module)

        for cls_name in selected_names:
            try:
                module_cls = getattr(module, cls_name)
                if (
                    isclass(module_cls)
                    and cls_name not in namespace
                    and required_base is not None
                    and issubclass(module_cls, required_base)
                ):
                    namespace[cls_name] = type(cls_name, (cls, module_cls), {})
                    # Update the class's module to match the caller's module
                    namespace[cls_name].__module__ = namespace["__name__"]
            except:
                cls.error(f"Error while creating class {cls_name} from module {module}")
                raise

    @property
    def model_extra(cls, **kwargs):
        """Override to give default {} if None"""
        extra = super().model_extra
        extra = {} if extra is None else extra
        return extra


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
