import torch

from ai.modality.modality import Modality


def test_modality_with_tensor():
    modality = Modality(torch.ones(3, 224, 224))

    assert isinstance(modality, Modality)

    modality = modality * 2

    assert modality.shape == (3, 224, 224)
    assert modality == torch.ones(3, 224, 224) * 2
    assert isinstance(modality, Modality)
