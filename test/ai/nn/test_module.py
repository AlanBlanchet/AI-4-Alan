from ai.nn.arch.resnet.resnet import ResNet18


def test_resnet18():
    model = ResNet18()

    assert len(list(model.parameters())) != 0
    assert len(model._modules) != 0

    # Check for buffers in the entire model hierarchy
    total_buffers = sum(len(module._buffers) for module in model.modules())
    assert total_buffers != 0

    assert len(model.layers) != 0

    assert model.input_proj is not None
