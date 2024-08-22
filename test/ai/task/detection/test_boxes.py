import torch
from ai.task.detection.box import nms


def test_nms():
    scores = torch.tensor([0.4, 0.3, 0.5])
    boxes = torch.tensor(
        [[0.1, 0.12, 0.5, 0.7], [0.9, 1.13, 0.51, 0.72], [0.2, 0.3, 1, 1]]
    )

    nms_scores, nms_boxes = nms(scores, boxes)

    assert nms_scores.shape == (2,)
    assert nms_boxes.shape == (2, 4)
