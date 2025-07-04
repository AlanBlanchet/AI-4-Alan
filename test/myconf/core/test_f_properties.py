from myconf import F, MyConf


def test_f_property_computation(function_instance):
    assert function_instance.computed == "computed_example"


def test_f_property_creation_with_base():
    from test.conftest import FunctionClass

    instance1 = FunctionClass(base="first")
    instance2 = FunctionClass(base="second")
    assert instance1.computed == "computed_first"
    assert instance2.computed == "computed_second"


def test_f_property_computation_during_init():
    class ComputedClass(MyConf):
        base: str = "test"
        call_count: int = 0

        def _increment_and_get(self):
            self.call_count += 1
            return self.call_count

        computed: str = F(lambda self: f"called_{self._increment_and_get()}")

    instance = ComputedClass()
    # F() properties are computed during initialization
    assert instance.call_count == 1
    first_call = instance.computed
    assert first_call == "called_1"
    # Subsequent accesses return the same computed value
    second_call = instance.computed
    assert second_call == "called_1"
    assert instance.call_count == 1  # Still only called once
