import torch

from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


def test_resnet18():
    model = ResNet18()

    sample = torch.randn(1, 3, 224, 224)

    out = model(sample)

    assert out.shape == (1, 512)


def test_resnet34():
    model = ResNet34()

    sample = torch.randn(1, 3, 224, 224)

    out = model(sample)

    assert out.shape == (1, 512)


def test_resnet50():
    model = ResNet50()

    sample = torch.randn(1, 3, 224, 224)

    out = model(sample)

    assert out.shape == (1, 2048)


def test_resnet101():
    model = ResNet101()

    sample = torch.randn(1, 3, 224, 224)

    out = model(sample)

    assert out.shape == (1, 2048)


def test_resnet152():
    model = ResNet152()

    sample = torch.randn(1, 3, 224, 224)

    out = model(sample)

    assert out.shape == (1, 2048)
