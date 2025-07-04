def test_basic_display(simple_instance):
    display = str(simple_instance)
    # name="test" is the default value, so it should be hidden
    assert "name=test" not in display
    # value=100 is different from default 42, so it should show
    assert "value=100" in display


def test_nested_display(complex_instance):
    display = str(complex_instance)
    assert "ComplexClass" in display
    assert "NestedClass" in display
    assert "x=100" in display


def test_structure():
    from test.conftest import IndentationClass, NestedClass

    instance = IndentationClass(nested=NestedClass(x=1, y="test"))
    display = str(instance)
    assert "IndentationClass" in display
    assert "NestedClass" in display
