from pytest import fixture

from .label_map import LabelMap


@fixture
def labels():
    return ["background", "person", "car", "dog"]


def test_label_map(labels):
    label_map = LabelMap(labels=labels)

    bg_id = label_map["background"]

    assert label_map[bg_id] == "background"
    assert len(label_map) == 4
