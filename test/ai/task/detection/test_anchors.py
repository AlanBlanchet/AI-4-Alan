import torch
from ai.task import AnchorManager
from pytest import fixture


@fixture
def manager():
    return AnchorManager([38, 19, 10, 5, 3, 1], [4, 6, 6, 6, 4, 4])


def test_anchor_ssd(manager):
    assert manager.anchors.shape == (8732, 4)


def test_anchor_ssd_encode(manager):
    gt_boxes = torch.tensor([[0.25, 0.25, 0.5, 0.5], [0, 0, 0, 0]])
    gt_labels = torch.tensor([1, 0])
    encoded = manager.match(gt_boxes, gt_labels)

    boxes, labels = encoded["boxes"], encoded["labels"]

    assert boxes.shape == (8732, 4)
    assert labels.shape == (8732,)

    decoded = manager.decode(boxes)
    assert torch.allclose(decoded[-1], gt_boxes[0])
