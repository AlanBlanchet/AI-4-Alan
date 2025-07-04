import inspect


def create_ide_signature(cls) -> inspect.Signature:
    """
    Create an IDE-friendly signature that shows proper parameter ordering
    based on the class's __new__ method signature.
    """
    properties = getattr(cls, "_myconf_properties", {})

    # Start with self parameter
    parameters = []

    # Get parameter order from __new__ method if available
    parameter_order = []
    new_required_params = set()  # Parameters required by __new__

    # Look for custom __new__ method in the MRO
    custom_new_method = None
    for base_cls in cls.__mro__:
        if (
            hasattr(base_cls, "__new__")
            and base_cls.__new__ != object.__new__
            and getattr(base_cls.__new__, "__self__", None) is not object
        ):
            custom_new_method = base_cls.__new__
            break

    if custom_new_method:
        try:
            new_sig = inspect.signature(custom_new_method)
            # Get parameter names from __new__, skipping 'cls'
            new_params = list(new_sig.parameters.values())[1:]

            for param in new_params:
                # Track which parameters are required by __new__
                if param.default == inspect.Parameter.empty:
                    new_required_params.add(param.name)

                # Map __new__ parameters to properties that exist and have init=True
                if (
                    param.name in properties
                    and properties[param.name].init
                    and not param.name.startswith("_")
                ):
                    parameter_order.append(param.name)
        except Exception:
            pass

    # If no custom __new__ or couldn't extract params, use declaration order
    if not parameter_order:
        # Respect declaration order by using class annotations order
        annotations = getattr(cls, "__annotations__", {})
        ordered_prop_names = list(annotations.keys())

        # Add any properties not in annotations (from parent classes) at the end
        for name in properties.keys():
            if name not in ordered_prop_names and not name.startswith("_"):
                ordered_prop_names.append(name)

        for name in ordered_prop_names:
            if (
                name in properties
                and not name.startswith("_")
                and properties[name].init
            ):
                parameter_order.append(name)

    # Add any properties not yet in parameter_order
    annotations = getattr(cls, "__annotations__", {})
    for name in annotations.keys():
        if (
            name in properties
            and not name.startswith("_")
            and properties[name].init
            and name not in parameter_order
        ):
            parameter_order.append(name)

    # Separate required and optional parameters
    required_params = []
    optional_params = []

    for name in parameter_order:
        if name not in properties:
            continue

        info = properties[name]

        # Skip private properties and invalid names
        if name.startswith("_") or not hasattr(info, "annotation"):
            continue

        # Skip fields with init=False - they shouldn't appear in signatures
        if hasattr(info, "init") and not info.init:
            continue

        # Include consumed parameters if they're required by __new__
        is_consumed = getattr(info, "is_consumed", False)
        is_required_by_new = name in new_required_params

        if is_consumed and not is_required_by_new:
            continue

        # Determine the parameter type to show in signature
        if getattr(info, "is_cast", False):
            # For Cast (non-consumed), show the input type in signature
            param_type = getattr(info, "input_type", info.annotation)
        else:
            param_type = info.annotation

        # Skip ClassVar properties - they shouldn't be in signatures
        from myconf.utils import is_class_var

        if is_class_var(param_type):
            continue

        # Determine if parameter has a default value
        # For parameters required by __new__, they don't have defaults in the signature
        if is_required_by_new:
            default_value = inspect.Parameter.empty
        else:
            # Check for both traditional values and F() functions
            has_default = info.value is not None or info.fn is not None

            # For F() properties, they should be treated as optional with a meaningful default
            if has_default:
                if info.fn is not None:
                    default_value = "F(...)"  # Show as F(...) for computed properties
                else:
                    default_value = info.value  # Use actual default value
            else:
                default_value = inspect.Parameter.empty

        param = inspect.Parameter(
            name=name,
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default_value,
            annotation=param_type,
        )

        # Separate into required vs optional
        if default_value == inspect.Parameter.empty:
            required_params.append(param)
        else:
            optional_params.append(param)

    # Combine parameters with required first, then optional
    parameters.extend(required_params)
    parameters.extend(optional_params)

    # Create and return the signature
    return inspect.Signature(parameters)


def apply_ide_signature(cls):
    """
    Apply the IDE-friendly signature to a class.
    This sets both __signature__ and __init__.__signature__ to ensure
    IDEs show the correct signature.
    """
    # Generate the signature
    class_sig = create_ide_signature(cls)

    # Set class signature (what IDEs see for class instantiation)
    cls.__signature__ = class_sig

    # Also set __init__ signature for completeness
    if hasattr(cls, "__init__"):
        # __init__ signature should include 'self' parameter
        init_params = [
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        init_params.extend(class_sig.parameters.values())
        init_sig = inspect.Signature(init_params)
        cls.__init__.__signature__ = init_sig

    return cls


def get_real_ide_signature(cls) -> str:
    """
    Get the signature string that IDEs actually see and display.
    This is useful for debugging signature issues.
    """
    if hasattr(cls, "__signature__"):
        return str(cls.__signature__)
    else:
        return str(inspect.signature(cls))


def debug_signature_info(cls):
    """
    Debug function to show all signature-related information for a class.
    """
    print(f"=== Signature Debug Info for {cls.__name__} ===")

    # Check for __signature__ attribute
    if hasattr(cls, "__signature__"):
        print(f"Class __signature__: {cls.__signature__}")
    else:
        print("No class __signature__ attribute")

    # Check inspect.signature on class
    try:
        class_sig = inspect.signature(cls)
        print(f"inspect.signature(class): {class_sig}")
    except Exception as e:
        print(f"Error getting class signature: {e}")

    # Check __init__ signature
    if hasattr(cls, "__init__"):
        try:
            init_sig = inspect.signature(cls.__init__)
            print(f"inspect.signature(__init__): {init_sig}")
        except Exception as e:
            print(f"Error getting __init__ signature: {e}")

    # Show parameter details
    try:
        sig = inspect.signature(cls)
        print("Parameter details:")
        for name, param in sig.parameters.items():
            print(
                f"  {name}: kind={param.kind.name}, default={param.default}, annotation={param.annotation}"
            )
    except Exception as e:
        print(f"Error getting parameter details: {e}")

    # Show Cast/Consumed info
    properties = getattr(cls, "_myconf_properties", {})
    cast_consumed = [
        (name, info)
        for name, info in properties.items()
        if getattr(info, "is_cast", False) or getattr(info, "is_consumed", False)
    ]
    if cast_consumed:
        print("Cast/Consumed properties:")
        for name, info in cast_consumed:
            if getattr(info, "is_cast", False):
                print(
                    f"  {name}: Cast[{getattr(info, 'input_type', '?')}, {getattr(info, 'output_type', '?')}]"
                )
            elif getattr(info, "is_consumed", False):
                print(f"  {name}: Consumed[{getattr(info, 'input_type', '?')}]")


def debug_ide_type_info(cls, instance=None):
    """
    Comprehensive diagnostic function to show what IDEs and different tools see.
    This goes beyond inspect to show what LSP servers and type checkers see.
    """
    print(f"\n=== IDE Type Diagnostic for {cls.__name__} ===")

    # 1. What __annotations__ shows (what IDEs read for type hints)
    print("1. Class __annotations__ (what IDEs see for type hints):")
    annotations = getattr(cls, "__annotations__", {})
    for name, annotation in annotations.items():
        print(f"  {name}: {annotation}")
        # Show the fully qualified name that type checkers see
        if hasattr(annotation, "__module__") and hasattr(annotation, "__name__"):
            print(
                f"    -> Fully qualified: {annotation.__module__}.{annotation.__name__}"
            )

    # 2. What inspect shows (Python runtime info)
    print("\n2. Inspect signature (Python runtime):")
    try:
        sig = inspect.signature(cls)
        for name, param in sig.parameters.items():
            if name != "self":
                print(f"  {name}: {param.annotation} = {param.default}")
    except Exception as e:
        print(f"  Error: {e}")

    # 3. MyConf internal property info
    print("\n3. MyConf property processing:")
    if hasattr(cls, "_myconf_properties"):
        for name, info in cls._myconf_properties.items():
            if not name.startswith("_"):
                cast_info = ""
                if getattr(info, "is_cast", False):
                    input_type = getattr(info, "input_type", "?")
                    output_type = getattr(info, "output_type", "?")
                    cast_info = f" [Cast: {input_type} -> {output_type}]"
                elif getattr(info, "is_consumed", False):
                    cast_info = " [Consumed]"
                print(f"  {name}: {info.annotation}{cast_info}")

    # 4. Runtime instance type info (if instance provided)
    if instance:
        print("\n4. Runtime instance types:")
        for name in annotations.keys():
            if hasattr(instance, name) and not name.startswith("_"):
                attr_value = getattr(instance, name)
                attr_type = type(attr_value)
                print(f"  instance.{name}: {attr_type}")

                # Show available methods for IDE autocomplete simulation
                if hasattr(attr_value, "__class__"):
                    public_methods = [
                        m for m in dir(attr_value) if not m.startswith("_")
                    ]
                    if public_methods:
                        print(
                            f"    -> Available methods: {public_methods[:5]}{'...' if len(public_methods) > 5 else ''}"
                        )

    # 5. Type checker perspective (simulate what pyright/mypy see)
    print("\n5. Type checker perspective:")
    print("   The following is what language servers/type checkers should see when")
    print("   you type 'instance.field_name.' and press Ctrl+Space:")

    for name, annotation in annotations.items():
        if not name.startswith("_"):
            print(f"   {name} -> {annotation}")
            if hasattr(annotation, "__dict__") or hasattr(annotation, "__class__"):
                # Try to get methods that would appear in autocomplete
                try:
                    if hasattr(annotation, "__origin__"):  # Generic types
                        origin = annotation.__origin__
                        methods = [
                            m
                            for m in dir(origin)
                            if not m.startswith("_")
                            and callable(getattr(origin, m, None))
                        ]
                    else:
                        methods = [
                            m
                            for m in dir(annotation)
                            if not m.startswith("_")
                            and callable(getattr(annotation, m, None))
                        ]

                    if methods:
                        print(
                            f"     IDE should suggest: {methods[:3]}{'...' if len(methods) > 3 else ''}"
                        )
                except:
                    pass

    # 6. Original annotation before MyConf processing
    print("\n6. Original annotations (before MyConf processing):")
    # Try to find the original Cast annotations in MRO
    for base in cls.__mro__:
        if hasattr(base, "__annotations__") and base is not cls:
            base_annotations = base.__annotations__
            if base_annotations:
                print(f"   From {base.__name__}: {base_annotations}")
                break
