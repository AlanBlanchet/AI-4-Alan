from myconf import F
from test.conftest import ChildClass, FBaseClass, FChildClass


def test_property_inheritance():
    instance = ChildClass()
    assert instance.base_field == "base"
    assert instance.child_field == "child"
    assert instance.shared_field == "child_value"


def test_property_merging(property_test_data):
    data = property_test_data
    assert data["merged_expected"]["name"] == "child"
    assert data["merged_expected"]["value"] == 1


def test_f_property_inheritance():
    instance = FChildClass()
    assert instance.computed_base == "base_test"
    assert instance.computed_child == "child_child"


def test_f_property_override():
    class OverrideClass(FBaseClass):
        computed_base: str = F(lambda self: f"override_{self.base}")

    instance = OverrideClass()
    assert instance.computed_base == "override_test"
