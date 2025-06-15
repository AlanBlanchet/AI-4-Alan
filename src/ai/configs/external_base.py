from functools import cached_property
from inspect import signature
from typing import Any, ClassVar

from pydantic import BaseModel

from .base import Base


class ExternalBase(Base):
    """Make any weird class compatible with Pydantic models.

    A 'weird' class is a class with class annotations for properties, functions as properties
    or even manually defined properties in method others than the `__init__`
    """

    XT_REMOVE_KEYS: ClassVar[list[str]] = []
    """Keys to be removed from external class"""
    XT_ADD_KEYS: ClassVar[list[str]] = []
    """Keys to be added to the external class. Usually the properties that are  hidden in the external __init__"""

    _XT_SPECIAL_KEYS: ClassVar[list[str]]
    """To make a class compatible with pydantic, we need to get the name of attributes
    that are not part of the pydantic model and used inside their __init__ fn.
    """

    @classmethod
    def _compute_external(cls):
        after_base_model = None
        for bcls in cls.__mro__:
            if after_base_model:
                return bcls

            if bcls == BaseModel:
                after_base_model = True

        raise ValueError(
            f"Cannot find any external __init__ cls class from {cls.__mro__}"
        )

    def __init__(self, **kwargs):
        init_kwargs = {}
        for k, v in signature(self.INIT_CLS.__init__).parameters.items():
            if k in kwargs:
                init_kwargs[k] = kwargs[k]
        # First initialize the INIT_CLS
        self.INIT_CLS.__init__(self, **init_kwargs)
        # Copy the current state (modules, buffers, parameters...)
        original_xt_state = self.__dict__.copy()

        # Initialize model
        # This erases the state but we need it in the post_init !
        # This is why we saved it in original_nn_state
        super().__init__(**kwargs)

        # Pydantic state
        pydantic_state = self.__dict__.copy()

        # Restore the state
        self.__dict__.update(original_xt_state)

        # We initialized pydantic
        self.pydantic_post_init(pydantic_state, original_xt_state)

        attrs = self.__dict__.copy()
        # Use the setattr method on INIT_CLS to use its logic
        for k, v in attrs.items():
            if isinstance(v, self.INIT_CLS):
                self.INIT_CLS.__setattr__(self, k, v)

        self.__dict__["__cached_properties__"] = {}
        for k in dir(self.__class__):
            v = getattr(self.__class__, k)
            if isinstance(v, cached_property):
                self.__cached_properties__[k] = v

        self.init()

    def __init_subclass__(cls, **kwargs):
        cls.INIT_CLS = cls._compute_external()

        # Some keys will never be handled by pydantic
        external_keys = dir(cls.INIT_CLS)

        # Add keys (usually the properties in the __init__ defined without type hints)
        for key in cls.XT_ADD_KEYS:
            if key not in external_keys:
                external_keys.append(key)

        # Remove keys (sometimes methods are used as properties ex: forward method in torch)
        for key in cls.XT_REMOVE_KEYS:
            if key in external_keys:
                external_keys.remove(key)

        cls._XT_SPECIAL_KEYS = external_keys

        if not cls.model_post_init.__module__.startswith("pydantic"):
            # User has defined a model_post_init method
            raise ValueError(
                f"model_post_init is a reserved method for Pydantic Modules {cls.__name__}. Please use the init instead."
            )
        return super().__init_subclass__(**kwargs)

    @cached_property
    def _init_cls_mro_annotations(self):
        annots = []
        for cls in self.INIT_CLS.__mro__[:-1]:  # Remove object
            annots.extend(cls.__annotations__.keys())
        return annots

    def pydantic_post_init(self, pydantic_state, xt_state):
        """Calling one init erases the __dict__ of the instance

        Here you have access to both initialized state dicts
        """
        ...

    def _in_init_cls_mro_annotations(self, name: str):
        return name in self._init_cls_mro_annotations

    def should_go_to_xt_state(self, name: str, value: Any):
        """Weither we should use the xt get/set or the pydantic one"""
        return False

    def __setattr__(self, name, value):
        try:
            if (
                name in self._XT_SPECIAL_KEYS
                or self.should_go_to_xt_state(name, value)
                or self._in_init_cls_mro_annotations(name)
            ):
                self.INIT_CLS.__setattr__(self, name, value)
            else:
                super().__setattr__(name, value)
        except Exception as e:
            raise Exception(
                f"Error while trying to set '{name}'. Maybe '{name}' needs to be added to the special keys to work with pydantic ?"
            ) from e

    def __getattr__(self, name):
        try:
            return self.INIT_CLS.__getattr__(self, name)
        except AttributeError as init_cls_e:
            try:
                return super().__getattr__(name)
            except AttributeError as pd_e:
                # Handle the cached properties
                super_dict = super().__dict__
                if "__cached_properties__" in super_dict:
                    cached = super_dict["__cached_properties__"]
                    if name in cached:
                        return cached.__get__(self)

                raise pd_e from init_cls_e

    def init(self): ...
