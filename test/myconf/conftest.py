"""
Global test fixtures and utilities for MyConf test suite.
"""

from pathlib import Path

import pytest

from myconf import F, MyConf


class SimpleClass(MyConf):
    """Simple class for basic testing"""

    name: str = "test"
    value: int = 42
    enabled: bool = True


class NestedClass(MyConf):
    """Nested class for testing object composition"""

    x: int
    y: str


class ComplexClass(MyConf):
    """Complex class with nested structures"""

    path: Path
    nested: NestedClass


class DisplayClass(MyConf):
    """Class for testing display formatting"""

    name: str = "display"
    count: int = 5
    visible: bool = True


class IndentationClass(MyConf):
    """Class for testing indentation display"""

    level: int = 1
    nested: DisplayClass = F(lambda self: DisplayClass(name=f"level_{self.level}"))


class FunctionClass(MyConf):
    """Class demonstrating F() functionality"""

    base: str = "example"
    base_value: int = 10
    computed: str = F(lambda self: f"computed_{self.base}")

    formatted: str = F(lambda self: f"Value: {self.computed}")


# Inheritance test classes
class BaseClass(MyConf):
    """Base class for inheritance tests"""

    base_field: str = "base"
    shared_field: str = "base_value"


class ChildClass(BaseClass):
    """Child class for inheritance tests"""

    child_field: str = "child"
    shared_field: str = "child_value"


class FBaseClass(MyConf):
    """Base class with F() properties"""

    base: str = "test"
    computed_base: str = F(lambda self: f"base_{self.base}")


class FChildClass(FBaseClass):
    """Child class with F() properties"""

    child: str = "child"
    computed_child: str = F(lambda self: f"child_{self.child}")


# Test configuration fixtures that tests expect
class A(MyConf):
    """Test configuration A"""

    value: int = 10
    name: str = "config_a"


class B(MyConf):
    """Test configuration B"""

    enabled: bool = True
    count: int = 5


class C(MyConf):
    """Test configuration C with nested objects"""

    a_config: A = F(lambda self: A())
    b_config: B = F(lambda self: B())


@pytest.fixture
def simple_instance():
    """Simple MyConf instance"""
    return SimpleClass(name="test", value=100)


@pytest.fixture
def function_instance():
    """Instance with F() functions"""
    return FunctionClass(base="example", base_value=15)


@pytest.fixture
def complex_instance():
    """Complex instance with nested objects"""
    return ComplexClass(path="test.txt", nested=NestedClass(x=100, y="nested"))


@pytest.fixture
def inheritance_instance():
    """Instance for testing inheritance"""
    return ChildClass()


@pytest.fixture
def sample_config():
    """Sample configuration dictionary"""
    return {"path": "sample.txt", "nested": {"x": 42, "y": "test"}}


@pytest.fixture
def a_inst():
    """Instance of configuration A"""
    return A(value=20, name="test_a")


@pytest.fixture
def a_inst2():
    """Second instance of configuration A"""
    return A(value=30, name="test_a2")


@pytest.fixture
def c_inst():
    """Instance of configuration C"""
    return C()


@pytest.fixture
def a_config():
    """Configuration dictionary for A"""
    return {"value": 25, "name": "from_dict"}


@pytest.fixture
def property_test_data():
    """Data for testing property inheritance and merging"""
    return {
        "base_props": {"name": "base", "value": 1},
        "child_props": {"name": "child", "extra": 2},
        "merged_expected": {"name": "child", "value": 1, "extra": 2},
    }
