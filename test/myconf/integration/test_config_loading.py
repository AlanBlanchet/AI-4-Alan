from pathlib import Path


def test_config_creation(sample_config):
    from test.conftest import ComplexClass

    instance = ComplexClass.from_config(sample_config)
    assert instance.path == Path("sample.txt")
    assert instance.nested.x == 42
    assert instance.nested.y == "test"


def test_basic_instantiation():
    from test.conftest import ComplexClass, NestedClass

    instance = ComplexClass(path="test.txt", nested=NestedClass(x=100, y="nested"))
    assert instance.path == Path("test.txt")
    assert instance.nested.x == 100


def test_nested_config_loading():
    from test.conftest import ComplexClass

    config = {"path": "/tmp/test.txt", "nested": {"x": 999, "y": "loaded_value"}}
    instance = ComplexClass.from_config(config)
    assert instance.path == Path("/tmp/test.txt")
    assert instance.nested.x == 999
    assert instance.nested.y == "loaded_value"
