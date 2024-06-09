import torch
from torchvision.ops.boxes import box_convert, box_iou


def _coco_xy_box_convert(boxes):
    return box_convert(boxes, "cxcywh", "xyxy")


batched_iou = torch.vmap(box_iou)
batched_box_convert = torch.vmap(_coco_xy_box_convert)


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    if boxes1.ndim > 3 or boxes2.ndim > 3:
        raise ValueError("Boxes should have at most 3 dimensions")

    if boxes1.ndim == 2:
        boxes1 = boxes1.unsqueeze(0)

    if boxes2.ndim == 2:
        boxes2 = boxes2.unsqueeze(0)

    ious = batched_iou(boxes1, boxes2)

    return ious
