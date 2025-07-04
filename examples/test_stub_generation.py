from pathlib import Path

from myconf import Cast, MyConf


class Input:
    def input_method(self):
        return "input"


class Output:
    def __init__(self, data=None):
        self.data = data

    def output_method(self):
        return "output"

    def batch_process(self):
        return "batch processing"


class TestClass(MyConf):
    field: Cast[Input, Output]
    regular_field: str = "test"


def generate_stub_for_cast_class(cls):
    """Generate a .pyi stub file for a class with Cast types"""
    module_name = cls.__module__
    if module_name == "__main__":
        # For test files, use the filename
        stub_path = Path("test_stub_generation.pyi")
    else:
        stub_path = Path(f"{module_name.replace('.', '/')}.pyi")

    stub_content = []

    # Add imports
    annotations = getattr(cls, "__annotations__", {})
    imported_types = set()

    for annotation in annotations.values():
        if hasattr(annotation, "__module__") and annotation.__module__ != "__main__":
            imported_types.add(
                f"from {annotation.__module__} import {annotation.__name__}"
            )

    for imp in sorted(imported_types):
        stub_content.append(imp)

    if imported_types:
        stub_content.append("")

    # Generate class definition
    stub_content.append(f"class {cls.__name__}:")

    # Process properties, showing output types for Cast
    properties = getattr(cls, "_myconf_properties", {})

    for name, info in properties.items():
        if name.startswith("_"):
            continue

        if getattr(info, "is_cast", False):
            # For Cast types, show the output type in the stub
            output_type = getattr(info, "output_type", None)
            if output_type:
                type_name = (
                    output_type.__name__
                    if hasattr(output_type, "__name__")
                    else str(output_type)
                )
                stub_content.append(f"    {name}: {type_name}")
        else:
            # For regular types, show as-is
            if hasattr(info, "annotation"):
                type_name = (
                    info.annotation.__name__
                    if hasattr(info.annotation, "__name__")
                    else str(info.annotation)
                )
                stub_content.append(f"    {name}: {type_name}")

    # Add constructor
    stub_content.append("    def __init__(self, **kwargs) -> None: ...")

    # Write stub file
    stub_content_str = "\n".join(stub_content)

    print(f"Generating stub file: {stub_path}")
    print("Stub content:")
    print(stub_content_str)
    print()

    with open(stub_path, "w") as f:
        f.write(stub_content_str)

    return stub_path


# Test the approach
print("=== TESTING STUB GENERATION FOR CAST TYPES ===")

# Generate stub file
stub_path = generate_stub_for_cast_class(TestClass)

# Show what IDEs should see vs what they actually see
print("What IDE should see after reading stub file:")
print("  instance.field -> Output (with methods: output_method, batch_process)")
print("  instance.regular_field -> str")

print(f"\nStub file created at: {stub_path}")
print("IDE should now provide Output methods when typing 'instance.field.'")

# Test runtime behavior still works
instance = TestClass(field=Input())
print("\nRuntime check:")
print(f"instance.field type: {type(instance.field)}")
print(f"instance.field.output_method(): {instance.field.output_method()}")

print(
    "\nTo test: Open this file in your IDE and type 'instance.field.' - you should see Output methods suggested"
)
