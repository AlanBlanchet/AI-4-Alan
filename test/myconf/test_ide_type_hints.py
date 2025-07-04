"""
Tests for IDE type hint support in MyConf.
Ensures that IDEs can properly infer types for MyConf properties.
"""

import inspect

from myconf import F, MyConf


class SimpleClass(MyConf):
    """Simple class for IDE type hint testing"""

    name: str = "default"
    value: int = 42
    enabled: bool = True


class FunctionClass(MyConf):
    """Class with F() properties for IDE testing"""

    base: str = "test"
    computed: str = F(lambda self: f"computed_{self.base}")


def test_class_annotations():
    """Test that class annotations are preserved"""
    assert SimpleClass.__annotations__["name"] == str
    assert SimpleClass.__annotations__["value"] == int
    assert SimpleClass.__annotations__["enabled"] == bool


def test_runtime_type_checking():
    """Test that runtime type conversion still works"""
    instance = SimpleClass(name="test", value="123", enabled="true")

    assert instance.name == "test"
    assert instance.value == 123
    assert instance.enabled is True
    assert isinstance(instance.value, int)
    assert isinstance(instance.enabled, bool)


def test_simple_myconf_signature():
    """Test that simple MyConf classes have proper IDE signatures"""

    class SimpleTest(MyConf):
        value: int = 42
        name: str = "test"

    # Check that signature shows proper parameters
    sig = inspect.signature(SimpleTest.__init__)
    params = list(sig.parameters.keys())

    assert "self" in params
    assert "value" in params
    assert "name" in params
    # No more **kwargs - signatures are now precise
    assert "kwargs" not in params

    # Check parameter details
    value_param = sig.parameters["value"]
    assert value_param.annotation == int
    assert value_param.default == 42

    name_param = sig.parameters["name"]
    assert name_param.annotation == str
    assert name_param.default == "test"

    # Check that parameters allow positional arguments (not keyword-only)
    assert value_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert name_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_class_signature_shows_all_properties():
    """Test that class signature includes all MyConf properties"""

    class TestClass(MyConf):
        required_prop: int
        optional_prop: str = "default"
        another_prop: float = 3.14

    sig = inspect.signature(TestClass.__init__)
    params = sig.parameters

    # Should have all properties as parameters
    assert "required_prop" in params
    assert "optional_prop" in params
    assert "another_prop" in params

    # Check types and defaults
    assert params["required_prop"].annotation == int
    assert params["required_prop"].default == inspect.Parameter.empty

    assert params["optional_prop"].annotation == str
    assert params["optional_prop"].default == "default"

    assert params["another_prop"].annotation == float
    assert params["another_prop"].default == 3.14


def test_no_generic_args_kwargs_signature():
    """Test that MyConf signatures are not generic (*args, **kwargs)"""

    class TestClass(MyConf):
        prop: int = 123

    # Get signature string representation
    sig_str = str(inspect.signature(TestClass.__init__))

    # Should NOT be generic
    assert sig_str != "(self, *args, **kwargs)"
    assert sig_str != "(*args, **kwargs)"

    # Should contain actual parameter names and types
    assert "prop" in sig_str
    assert "int" in sig_str
    assert "123" in sig_str


def test_signature_excludes_underscore_properties():
    """Test that underscore properties are excluded from signature"""

    class TestClass(MyConf):
        public_prop: int = 42
        _private_prop: str = "hidden"

    sig = inspect.signature(TestClass.__init__)
    params = list(sig.parameters.keys())

    assert "public_prop" in params
    assert "_private_prop" not in params


def test_cast_type_ide_annotations():
    """Test that Cast types show output_type in annotations for IDE support"""
    from myconf import Cast

    class InputType:
        def __init__(self, value: str = "test"):
            self.value = value

    class OutputType:
        def __init__(self, data):
            self.data = data

        def output_method(self):
            return "output method"

    class CastTestClass(MyConf):
        casted_field: Cast[InputType, OutputType]
        regular_field: InputType = InputType()

    # Class annotations should show OutputType, not Cast[InputType, OutputType]
    annotations = CastTestClass.__annotations__
    assert annotations["casted_field"] == OutputType
    assert annotations["regular_field"] == InputType

    # Signature should show InputType for the parameter
    sig = inspect.signature(CastTestClass.__init__)
    casted_param = sig.parameters["casted_field"]
    assert casted_param.annotation == InputType

    # Property info should have correct cast information
    properties = getattr(CastTestClass, "_myconf_properties", {})
    cast_prop = properties["casted_field"]
    assert getattr(cast_prop, "is_cast", False) == True
    assert getattr(cast_prop, "input_type", None) == InputType
    assert getattr(cast_prop, "output_type", None) == OutputType
    assert cast_prop.annotation == OutputType
