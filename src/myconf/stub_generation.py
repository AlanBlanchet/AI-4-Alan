"""Stub file generation for MyConf classes with Cast types"""

import inspect
from pathlib import Path
from typing import Set


def generate_stub_file(cls, output_path: Path = None) -> Path:
    """Generate a .pyi stub file for a MyConf class with proper Cast type handling"""

    if output_path is None:
        # Find the actual source file location
        try:
            source_file = inspect.getfile(cls)
            if source_file.endswith(".py"):
                # Create .pyi file next to the .py file
                output_path = Path(source_file).with_suffix(".pyi")
            else:
                # Fallback for edge cases
                output_path = Path(f"{cls.__name__.lower()}.pyi")
        except (TypeError, OSError):
            # Fallback for classes defined in REPL or special cases
            module_name = cls.__module__
            if module_name == "__main__":
                output_path = Path(f"{cls.__name__.lower()}.pyi")
            else:
                output_path = Path(f"{module_name.replace('.', '/')}.pyi")

    stub_content = []
    imported_types = set()
    local_types = set()

    # Get class properties
    properties = getattr(cls, "_myconf_properties", {})

    # Collect types that need to be imported or defined
    for name, info in properties.items():
        if name.startswith("_"):
            continue

        target_type = None
        if getattr(info, "is_cast", False):
            target_type = getattr(info, "output_type", None)
        elif hasattr(info, "annotation"):
            target_type = info.annotation

        if target_type:
            _collect_type_imports(
                target_type, imported_types, local_types, cls.__module__
            )

    # Add imports
    for imp in sorted(imported_types):
        stub_content.append(imp)

    if imported_types:
        stub_content.append("")

    # Add local type definitions
    for type_def in sorted(local_types):
        stub_content.append(type_def)

    if local_types:
        stub_content.append("")

    # Generate class definition
    stub_content.append(f"class {cls.__name__}:")

    # Add properties
    for name, info in properties.items():
        if name.startswith("_"):
            continue

        if getattr(info, "is_cast", False):
            output_type = getattr(info, "output_type", None)
            if output_type:
                type_name = _get_type_name(output_type)
                stub_content.append(f"    {name}: {type_name}")
        else:
            if hasattr(info, "annotation") and info.annotation:
                type_name = _get_type_name(info.annotation)
                stub_content.append(f"    {name}: {type_name}")

    # Add constructor
    stub_content.append("    def __init__(self, **kwargs) -> None: ...")

    # Write stub file
    stub_content_str = "\n".join(stub_content)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(stub_content_str)

    return output_path


def _get_type_name(type_obj) -> str:
    """Get a proper type name for stub files"""
    if hasattr(type_obj, "__name__"):
        return type_obj.__name__
    elif hasattr(type_obj, "__origin__"):
        # Handle generic types like List[int]
        origin = type_obj.__origin__
        args = getattr(type_obj, "__args__", ())
        if args:
            arg_names = [_get_type_name(arg) for arg in args]
            return f"{origin.__name__}[{', '.join(arg_names)}]"
        else:
            return origin.__name__
    else:
        return str(type_obj)


def _collect_type_imports(
    type_obj, imported_types: Set[str], local_types: Set[str], current_module: str
):
    """Collect imports and local type definitions needed for a type"""
    if type_obj is None:
        return

    # Handle basic types
    if hasattr(type_obj, "__module__"):
        module = type_obj.__module__
        name = getattr(type_obj, "__name__", str(type_obj))

        if module == "builtins":
            # Built-in types don't need imports
            pass
        elif module == current_module or module == "__main__":
            # Local type - need to define it in stub
            if hasattr(type_obj, "__annotations__") or inspect.isclass(type_obj):
                local_types.add(_generate_class_stub(type_obj))
        else:
            # External type - need import
            imported_types.add(f"from {module} import {name}")

    # Handle generic types
    if hasattr(type_obj, "__args__"):
        for arg in type_obj.__args__:
            _collect_type_imports(arg, imported_types, local_types, current_module)


def _generate_class_stub(cls) -> str:
    """Generate a stub definition for a class"""
    lines = [f"class {cls.__name__}:"]

    # Add methods that are likely to be used by IDEs
    if hasattr(cls, "__init__"):
        lines.append("    def __init__(self, *args, **kwargs) -> None: ...")

    # Add public methods
    for name in dir(cls):
        if not name.startswith("_") and callable(getattr(cls, name, None)):
            lines.append(f"    def {name}(self, *args, **kwargs) -> Any: ...")

    if len(lines) == 1:
        lines.append("    ...")

    return "\n".join(lines)


def auto_generate_stubs(cls):
    """Automatically generate stub files for a MyConf class"""
    try:
        stub_path = generate_stub_file(cls)
        return stub_path
    except Exception:
        # Silently fail - stub generation is optional
        return None
