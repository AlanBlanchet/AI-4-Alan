import torch


def decode(boxes: list[torch.Tensor], num_defaults: list[int], num_classes: int):
    new_boxes = []
    for cell_boxes, defaults in zip(boxes, num_defaults):
        B, *_ = cell_boxes.shape
        anchor_boxes = cell_boxes.view(B, -1, defaults, num_classes + 4)
        new_boxes.append(anchor_boxes.flatten(-3, -2))

    new_boxes = torch.cat(new_boxes, dim=-2)

    conf, loc = new_boxes.split([num_classes, 4], dim=-1)

    return conf, loc
