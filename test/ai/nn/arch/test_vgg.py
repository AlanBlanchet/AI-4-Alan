import torch
from ai.nn.arch.vgg.vgg import VGG
from pytest import mark


@mark.model
def test_vgg_forward():
    model = VGG()

    input = torch.randn(1, 3, 224, 224)

    out = model(input)

    assert out.shape == (1, 512, 7, 7)
