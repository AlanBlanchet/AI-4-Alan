import torch

from ai.data.batch import Batch
from ai.modality.image.modality import ChannelData
from ai.nn.arch.resnet.resnet import ResNet18


def test_resnet18_auto_cast_single():
    resnet = ResNet18()
    resnet.prepare()

    single_data = ChannelData(torch.randn(3, 224, 224))
    result = resnet.classify(single_data)

    assert result.shape[0] == 1
    assert len(result.shape) == 2


def test_resnet18_auto_cast_batch():
    resnet = ResNet18()
    resnet.prepare()

    batch_data = Batch.collate_fn(
        [ChannelData(torch.randn(3, 224, 224)), ChannelData(torch.randn(3, 224, 224))]
    )
    result = resnet.classify(batch_data)

    assert result.shape[0] == 2
    assert len(result.shape) == 2
