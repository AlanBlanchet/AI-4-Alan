"""Property processing for MyConf"""

import sys
from functools import cached_property
from typing import get_args, get_origin

from .constants import MYCONF_PROPERTIES_ATTR
from .core import Cast, Consumed, PropertyInfo
from .utils import is_class_var


def _analyze_cast_consumed_type(annotation):
    """Analyze annotation to detect Cast or Consumed types"""
    origin = get_origin(annotation)

    if origin is Cast:
        args = get_args(annotation)
        if len(args) == 2:
            input_type, output_type = args

            # Recursively analyze input type for nested Casts
            nested_casts = []
            if get_origin(input_type) is not None:  # Union, List, etc.
                from typing import Union

                if get_origin(input_type) is Union:
                    union_args = get_args(input_type)
                    for union_arg in union_args:
                        if get_origin(union_arg) is Cast:
                            nested_analysis = _analyze_cast_consumed_type(union_arg)
                            if nested_analysis and nested_analysis.get("is_cast"):
                                nested_casts.append(nested_analysis)

            return {
                "is_cast": True,
                "input_type": input_type,
                "output_type": output_type,
                "nested_casts": nested_casts,
                "converter": None,  # Will use default converter
            }
    elif origin is Consumed:
        args = get_args(annotation)
        if len(args) == 1:
            # Check if the inner type is also Cast
            inner_type = args[0]
            inner_analysis = _analyze_cast_consumed_type(inner_type)
            if inner_analysis and inner_analysis.get("is_cast"):
                # Combine Consumed + Cast
                return {
                    "is_consumed": True,
                    "is_cast": True,
                    "input_type": inner_analysis["input_type"],
                    "output_type": inner_analysis["output_type"],
                    "nested_casts": inner_analysis.get("nested_casts", []),
                    "converter": inner_analysis.get("converter"),
                }
            else:
                # Just Consumed
                return {
                    "is_consumed": True,
                    "input_type": inner_type,
                    "output_type": None,
                }

    return None


def setup_myconf_properties(cls):
    """Set up _myconf_properties for a class"""
    if hasattr(cls, MYCONF_PROPERTIES_ATTR) and cls._myconf_properties:
        own_annotations = getattr(cls, "__annotations__", {})
        if not own_annotations:
            return

    parent_properties = {}
    for base in cls.__mro__[1:]:
        if hasattr(base, MYCONF_PROPERTIES_ATTR):
            for prop_name, prop_info in base._myconf_properties.items():
                if prop_name not in parent_properties:
                    parent_properties[prop_name] = prop_info

    properties = parent_properties.copy()
    annotations = getattr(cls, "__annotations__", {})

    # Process class attributes that might override parent properties
    class_overrides = {}
    for prop_name, attr_value in cls.__dict__.items():
        if isinstance(attr_value, cached_property) or isinstance(
            attr_value, PropertyInfo
        ):
            continue
        if prop_name in parent_properties:
            class_overrides[prop_name] = attr_value

    # Track annotations that need to be updated for IDE support
    updated_annotations = annotations.copy()

    # Process annotated properties
    for prop_name, annotation in annotations.items():
        if is_class_var(annotation):
            continue

        if isinstance(annotation, str):
            cls_module = sys.modules.get(cls.__module__)
            if cls_module and hasattr(cls_module, annotation):
                annotation = getattr(cls_module, annotation)

        # Check for Cast/Consumed types
        cast_consumed_info = _analyze_cast_consumed_type(annotation)

        prop_info = None
        if prop_name in cls.__dict__:
            attr_value = cls.__dict__[prop_name]
            if isinstance(attr_value, cached_property):
                continue
            if isinstance(attr_value, PropertyInfo):
                prop_info = PropertyInfo(
                    fn=attr_value.fn,
                    value=attr_value.value,
                    annotation=annotation,
                    init=attr_value.init,
                )
            else:
                prop_info = PropertyInfo(value=attr_value, annotation=annotation)
        elif prop_name in class_overrides:
            attr_value = class_overrides[prop_name]
            prop_info = PropertyInfo(value=attr_value, annotation=annotation)
        elif prop_name in parent_properties:
            parent_info = parent_properties[prop_name]
            prop_info = PropertyInfo(
                value=parent_info.value,
                fn=parent_info.fn,
                annotation=annotation,
                init=parent_info.init,
            )
        else:
            prop_info = PropertyInfo(annotation=annotation)

        # Apply Cast/Consumed info if detected
        if cast_consumed_info:
            for key, value in cast_consumed_info.items():
                setattr(prop_info, key, value)

            # Consumed parameters don't get stored as attributes
            if cast_consumed_info.get("is_consumed"):
                prop_info.init = True  # Still part of init signature
                # Don't set the attribute on the class
                # Remove consumed parameters from annotations for IDE
                if prop_name in updated_annotations:
                    del updated_annotations[prop_name]
            else:
                # Cast types get stored with their output type
                output_type = cast_consumed_info.get("output_type", annotation)
                prop_info.annotation = output_type
                # Update annotations for IDE support - show output type instead of Cast type
                updated_annotations[prop_name] = output_type

        properties[prop_name] = prop_info

    # Update class annotations for IDE support
    cls.__annotations__ = updated_annotations

    # Process remaining class overrides that don't have annotations
    for prop_name, attr_value in class_overrides.items():
        if prop_name not in annotations and prop_name in parent_properties:
            parent_info = parent_properties[prop_name]
            prop_info = PropertyInfo(
                value=attr_value,
                fn=parent_info.fn,
                annotation=parent_info.annotation,
                init=parent_info.init,
            )
            properties[prop_name] = prop_info

    # Process PropertyInfo instances directly on the class
    for prop_name, attr_value in cls.__dict__.items():
        if isinstance(attr_value, cached_property):
            continue
        if isinstance(attr_value, PropertyInfo) and prop_name not in properties:
            properties[prop_name] = attr_value

    cls._myconf_properties = properties

    for prop_name, prop_info in properties.items():
        if isinstance(prop_info, PropertyInfo):
            # Don't set consumed properties as class attributes at all
            if getattr(prop_info, "is_consumed", False):
                # Remove any existing attribute for consumed properties
                # Only delete if it's actually defined in this class, not inherited
                if prop_name in cls.__dict__:
                    delattr(cls, prop_name)
                continue

            if prop_info.value is not None and not prop_info.fn:
                setattr(cls, prop_name, prop_info.value)
            elif (
                hasattr(cls, prop_name)
                and prop_name in cls.__dict__
                and not prop_info.fn
            ):
                delattr(cls, prop_name)


def update_class_signatures(cls):
    """Update class signatures for better IDE support"""
    from .ide_signature import apply_ide_signature

    apply_ide_signature(cls)
