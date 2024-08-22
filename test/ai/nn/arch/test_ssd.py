import torch
from ai.nn.arch.resnet.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from ai.nn.arch.ssd.ssd import SSD
from ai.nn.arch.vgg.models import VGG13
from pytest import fixture, mark


@fixture
def input():
    return torch.randn(1, 3, 300, 300)


def assert_shapes(out):
    boxes = out["boxes"]
    scores = out["scores"]

    assert boxes.shape == (1, 8732, 4)
    assert scores.shape == (1, 8732, 10)


@mark.model
def test_ssd_vgg(input: torch.Tensor):
    model = SSD(num_classes=10)

    assert_shapes(model(input))


@mark.model
def test_ssd_vgg13(input: torch.Tensor):
    backbone = VGG13()
    model = SSD(backbone, num_classes=10)

    assert_shapes(model(input))


@mark.model
def test_ssd_resnet18(input: torch.Tensor):
    backbone = ResNet18()
    model = SSD(backbone, num_classes=10)

    assert_shapes(model(input))


@mark.model
def test_ssd_resnet34(input: torch.Tensor):
    backbone = ResNet34()
    model = SSD(backbone, num_classes=10)

    assert_shapes(model(input))


@mark.model
def test_ssd_resnet50(input: torch.Tensor):
    backbone = ResNet50()
    model = SSD(backbone, num_classes=10)

    assert_shapes(model(input))


@mark.model
def test_ssd_resnet101(input: torch.Tensor):
    backbone = ResNet101()
    model = SSD(backbone, num_classes=10)

    assert_shapes(model(input))


@mark.model
def test_ssd_resnet152(input: torch.Tensor):
    backbone = ResNet152()
    model = SSD(backbone, num_classes=10)

    assert_shapes(model(input))
