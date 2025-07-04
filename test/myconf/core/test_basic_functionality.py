from pathlib import Path


def test_basic_creation(simple_instance):
    assert simple_instance.name == "test"
    assert simple_instance.value == 100


def test_nested_objects(complex_instance):
    assert complex_instance.path == Path("test.txt")
    assert complex_instance.nested.x == 100
    assert complex_instance.nested.y == "nested"


def test_type_conversion():
    from test.conftest import NestedClass

    instance = NestedClass(x="123", y=456)
    assert instance.x == 123
    assert instance.y == "456"
