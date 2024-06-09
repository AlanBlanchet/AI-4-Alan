import torch
from pytest import fixture

from .anchor import AnchorManager


@fixture
def manager():
    return AnchorManager([38, 19, 10, 5, 3, 1], [4, 6, 6, 6, 4, 4])


def test_anchor_ssd(manager):
    assert manager.anchors.shape == (8732, 4)


def test_anchor_ssd_encode(manager):
    gt_boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5]])
    gt_labels = torch.tensor([1])
    encoded, labels, pos_mask = manager.encode(gt_boxes, gt_labels)

    decoded = manager.decode(encoded)

    assert encoded.shape == (8732, 4)
    assert torch.allclose(encoded[0], torch.tensor([0.5, 0.5, 0, 0]))
