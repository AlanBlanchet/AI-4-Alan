import torch
from torchvision.ops.boxes import box_convert, box_iou

batched_iou = torch.vmap(box_iou)
batched_box_convert_coco_to_xy = torch.vmap(lambda x: box_convert(x, "cxcywh", "xyxy"))
batched_box_convert_xy_to_coco = torch.vmap(lambda x: box_convert(x, "xyxy", "cxcywh"))
