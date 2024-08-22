from ai.grad import Value


def test_value_operations():
    x = Value(1)
    y = Value(2)

    assert x + y == Value(3)
    assert x + y == 3

    assert x - y == -1
    assert (x - y) * 5 == -5

    assert x * y == 2

    assert y / x == 2
    assert x // y == 0


def test_value_number_operations():
    x = Value(1)
    y = 2

    assert x + y == Value(3)
    assert x + y == 3

    assert x - y == -1
    assert (x - y) * 5 == -5

    assert x * y == 2

    assert y / x == 2
    assert x // y == 0


def test_number_value_operations():
    x = 1
    y = Value(2)

    assert x + y == Value(3)
    assert x + y == 3

    assert x - y == -1
    assert (x - y) * 5 == -5

    assert x * y == 2

    assert y / x == 2
    assert x // y == 0
