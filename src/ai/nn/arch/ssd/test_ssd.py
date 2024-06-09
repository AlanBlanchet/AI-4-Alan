import torch
from pytest import fixture, mark

from ..vgg.models import VGG13
from .ssd import SSD


@fixture
def input():
    return torch.randn(1, 3, 300, 300)


@mark.model
def test_ssd_vgg(input: torch.Tensor):
    model = SSD(num_classes=10)

    out = model(input)

    loc = out["location"]
    conf = out["confidence"]

    assert loc.shape == (1, 8732, 4)
    assert conf.shape == (1, 8732, 10)


@mark.model
def test_ssd_vgg13(input: torch.Tensor):
    backbone = VGG13()
    model = SSD(backbone, num_classes=10)

    out = model(input)

    loc = out["location"]
    conf = out["confidence"]

    assert loc.shape == (1, 8732, 4)
    assert conf.shape == (1, 8732, 10)
