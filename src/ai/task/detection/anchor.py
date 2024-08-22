from typing import NamedTuple

import torch
import torch.nn as nn

from .utils import (
    batched_box_convert_coco_to_xy,
    batched_box_convert_xy_to_coco,
    batched_iou,
)


class AnchorEncodeOutput(NamedTuple):
    encoded_boxes: torch.Tensor
    encoded_labels: torch.Tensor
    pos_mask: torch.Tensor


class AnchorManager(nn.Module):
    def __init__(
        self,
        feature_maps: list[int],
        num_anchors: list[int],
        scales=[0.2, 0.9],
        ratios=[1, 1, 2, 3, 1 / 2, 1 / 3],
        background_id=0,
    ):
        super().__init__()

        ratios = torch.tensor(ratios)
        self.num_anchors = num_anchors
        self.background_id = background_id

        self.scales = [torch.linspace(*scales, n) for n in num_anchors]
        self.ratios = [ratios[:n] for n in num_anchors]

        self.register_buffer("anchors_cxcywh", self._generate_anchors(feature_maps))
        self.register_buffer(
            "anchors_xyxy", batched_box_convert_coco_to_xy(self.anchors_cxcywh)
        )
        self.anchors_cxcywh: torch.Tensor
        self.anchors_xyxy: torch.Tensor

    def _generate_anchors(self, feature_maps):
        all_anchors = []

        for idx, (scales, ratios) in enumerate(zip(self.scales, self.ratios)):
            anchors = []
            size = feature_maps[idx]
            # For every cell in the feature map
            for i in range(size):
                for j in range(size):
                    cx = (j + 0.5) / size
                    cy = (i + 0.5) / size
                    for scale, ratio in zip(scales, ratios):
                        ratio_sqrt = ratio.sqrt()
                        w, h = scale * ratio_sqrt, scale / ratio_sqrt
                        anchors.append([cx, cy, w, h])
            all_anchors.append(torch.tensor(anchors))

        return torch.cat(all_anchors)

    def match(self, gt_labels: torch.Tensor, gt_boxes: torch.Tensor):
        eps = 1e-6
        device = gt_boxes.device
        anchors = self.anchors_cxcywh

        is_batched = gt_boxes.ndim > 2
        N = gt_boxes.shape[-2]

        # Support batched inputs
        exp = gt_boxes.shape[:-2] or (1,)
        exp_tuple = (*exp, -1, -1)
        anchors = anchors.expand(exp_tuple)
        anchors_xyxy = self.anchors_xyxy.expand(exp_tuple)
        gt_boxes = gt_boxes.expand(exp_tuple)
        gt_labels = gt_labels.expand(exp_tuple[:-1])

        # To coco
        gt_boxes_coco = batched_box_convert_xy_to_coco(gt_boxes)

        B = anchors.shape[0]

        # Encoded tensors for output
        encoded = torch.zeros_like(anchors, dtype=gt_boxes.dtype, device=device)
        encoded_labels = torch.full(
            anchors.shape[:-1],
            fill_value=self.background_id,
            dtype=torch.long,
            device=device,
        )

        # IoU with xyxy formats (B, A, Gt)
        ious = batched_iou(anchors_xyxy, gt_boxes).float()

        # Find the best ground truth box for each anchor (B, A)
        anchors_gt, anchors_gt_idx = ious.max(dim=-1)
        # Find the best anchor for each ground truth box (B, Gt)
        _, gt_anchors_idx = ious.max(dim=-2)

        # Required for batching
        # range_B = torch.arange(B, dtype=torch.long, device=device)[:, None]
        range_N = torch.arange(N, dtype=torch.long, device=device)
        # Match every ground truth to an anchor
        for b in range(B):
            batch_gt_anchors_idx = gt_anchors_idx[b]
            anchors_gt_idx[b, batch_gt_anchors_idx] = range_N
            anchors_gt[b, batch_gt_anchors_idx] = 2  # Ensure positive match

        # Filter out low IoU matches
        pos_mask = anchors_gt > 0.5

        for b in range(B):
            anchor = anchors[b]
            b_gt_boxes = gt_boxes_coco[b]
            b_gt_labels = gt_labels[b]
            b_anchors_gt_idx = anchors_gt_idx[b]
            b_pos_mask = pos_mask[b]
            b_encoded_labels = encoded_labels[b]

            # Set to gt_labels (0 is background)
            b_encoded_labels[b_pos_mask] = b_gt_labels[b_anchors_gt_idx[b_pos_mask]]

            # Encode bounding boxes
            matched_gtb = b_gt_boxes[b_anchors_gt_idx]

            # Relative anchor offsets
            cxy = [0, 1]
            wh = [2, 3]

            encoded[b, :, cxy] = (matched_gtb[:, cxy] - anchor[:, cxy]) / anchor[:, wh]
            encoded[b, :, wh] = (
                (matched_gtb[:, wh] + eps) / (anchor[:, wh] + eps)
            ).log()

        if not is_batched:
            # Return to original tensor shapes
            encoded = encoded.squeeze(0)
            encoded_labels = encoded_labels.squeeze(0)
            pos_mask = pos_mask.squeeze(0)

        return dict(
            encoded_labels=encoded_labels, encoded_boxes=encoded, pos_mask=pos_mask
        )

    def decode(self, encoded: torch.Tensor):
        device = encoded.device
        dtype = encoded.dtype
        anchors = self.anchors_cxcywh.to(device=device, dtype=dtype)

        decoded = torch.zeros_like(encoded, device=device, dtype=encoded.dtype)

        cxy = [0, 1]
        wh = [2, 3]

        decoded[..., cxy] = encoded[..., cxy] * anchors[..., wh] + anchors[..., cxy]
        decoded[..., wh] = (encoded[..., wh].exp() * anchors[..., wh]).to(anchors.dtype)

        return decoded
