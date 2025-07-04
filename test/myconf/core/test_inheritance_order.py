"""
Test that MyConf works correctly with different inheritance orders.
This is especially important for PyTorch compatibility.
"""

from myconf import MyConf


class MockParent:
    """Mock parent class that simulates a framework like PyTorch"""

    def __init__(self):
        self._framework_attr = "initialized"
        self.special_value = 42


class MyConfFirst(MyConf, MockParent):
    """Test MyConf first in inheritance order"""

    config_value: int = 10
    name: str = "myconf_first"


class ParentFirst(MockParent, MyConf):
    """Test parent first in inheritance order"""

    config_value: int = 20
    name: str = "parent_first"


def test_myconf_first_inheritance():
    """Test class with MyConf first in inheritance order"""
    instance = MyConfFirst()

    # MyConf properties should work
    assert instance.config_value == 10
    assert instance.name == "myconf_first"

    # Parent initialization should have run
    assert hasattr(instance, "_framework_attr")
    assert instance._framework_attr == "initialized"
    assert instance.special_value == 42

    # Type conversion should work
    instance.config_value = "30"
    assert instance.config_value == 30
    assert isinstance(instance.config_value, int)


def test_parent_first_inheritance():
    """Test class with parent first in inheritance order"""
    instance = ParentFirst()

    # MyConf properties should work
    assert instance.config_value == 20
    assert instance.name == "parent_first"

    # Parent initialization should have run
    assert hasattr(instance, "_framework_attr")
    assert instance._framework_attr == "initialized"
    assert instance.special_value == 42

    # Type conversion should work
    instance.config_value = "40"
    assert instance.config_value == 40
    assert isinstance(instance.config_value, int)


def test_both_inheritance_orders_equivalent():
    """Test that both inheritance orders provide equivalent functionality"""
    myconf_first = MyConfFirst()
    parent_first = ParentFirst()

    # Both should have parent framework attributes
    assert hasattr(myconf_first, "_framework_attr")
    assert hasattr(parent_first, "_framework_attr")

    # Both should have MyConf properties working
    assert hasattr(myconf_first, "config_value")
    assert hasattr(parent_first, "config_value")

    # Both should support type conversion
    myconf_first.config_value = "100"
    parent_first.config_value = "200"

    assert myconf_first.config_value == 100
    assert parent_first.config_value == 200
    assert isinstance(myconf_first.config_value, int)
    assert isinstance(parent_first.config_value, int)
