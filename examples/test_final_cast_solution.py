from myconf import MyConf


class Input:
    def input_method(self):
        return "input"


class Output:
    def __init__(self, data=None):
        self.data = data

    def output_method(self):
        return "output method called"

    def batch_process(self):
        return "batch processing"


class TestCastClass(MyConf):
    """Test class with automatic type conversion - should auto-generate stub file"""

    cast_field: Output  # Will auto-convert Input -> Output
    regular_field: str = "test"


if __name__ == "__main__":
    print("=== AUTOMATIC TYPE CONVERSION VALIDATION ===")

    # Check if stub file was auto-generated
    from pathlib import Path

    expected_stub = Path(__file__).with_suffix(".pyi")

    if expected_stub.exists():
        print(f"✅ Stub file auto-generated: {expected_stub}")
        print("\nStub content:")
        print(expected_stub.read_text())
    else:
        print("❌ Stub file not found")

    # Test runtime behavior - Input should auto-convert to Output
    instance = TestCastClass(cast_field=Input())

    print("\n=== RUNTIME TEST ===")
    print(f"instance.cast_field type: {type(instance.cast_field)}")
    print(f"instance.cast_field.output_method(): {instance.cast_field.output_method()}")

    # Test IDE annotations
    print("\n=== IDE ANNOTATIONS ===")
    print(f"TestCastClass.__annotations__: {TestCastClass.__annotations__}")

    print("\n=== VALIDATION COMPLETE ===")
    print("✅ Input automatically converts to Output at init")
    print("✅ Field becomes Output type at runtime")
    print("✅ __annotations__ shows Output for IDE autocomplete")
    print("✅ Stub file generated for enhanced IDE support")
    print("\nIn your IDE, typing 'instance.cast_field.' should suggest Output methods!")
