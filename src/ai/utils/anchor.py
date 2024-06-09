from typing import NamedTuple

import torch
import torch.nn as nn

from .iou import batched_box_convert, batched_iou


class AnchorEncodeOutput(NamedTuple):
    encoded_boxes: torch.Tensor
    encoded_labels: torch.Tensor
    pos_mask: torch.Tensor


class AnchorManager(nn.Module):
    def __init__(
        self,
        feature_maps: list[int],
        num_anchors: list[int],
        scales=[0.1, 0.9],
        ratios=[0.5, 2],
        background_id=0,
    ):
        super().__init__()

        self.num_anchors = num_anchors
        self.background_id = background_id

        self.scales = [
            torch.linspace(*scales, num_anchor) for num_anchor in num_anchors
        ]
        self.ratios = [
            torch.linspace(*ratios, num_anchor) for num_anchor in num_anchors
        ]

        self.anchors = self._generate_anchors(feature_maps)

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

    def encode(
        self, gt_boxes: torch.Tensor, gt_labels: torch.Tensor, mask: torch.Tensor
    ):
        eps = 1e-6
        device = gt_boxes.device
        anchors = self.anchors.to(device)

        is_batched = gt_boxes.ndim > 2
        N = gt_boxes.shape[-2]

        # Support batched inputs
        exp = gt_boxes.shape[:-2] or (1,)
        exp_tuple = (*exp, -1, -1)
        anchors = anchors.expand(exp_tuple)
        gt_boxes = gt_boxes.expand(exp_tuple)

        B = anchors.shape[0]

        encoded = torch.zeros(anchors.shape, dtype=torch.float32, device=device)
        encoded_labels = torch.full(
            anchors.shape[:-1],
            self.background_id,
            dtype=torch.long,
            device=device,
        )

        # To xyxy
        anchors_xyxy = batched_box_convert(anchors)
        # IoU with xyxy formats (B, A, Gt)
        ious = batched_iou(anchors_xyxy, gt_boxes).float()

        # Find the best ground truth box for each anchor (B, A)
        best_gt_iou, best_gt_idx = ious.max(dim=-1)

        # Find the best anchor for each ground truth box (B, Gt)
        _, best_anchor_idx = ious.max(dim=-2)
        for b in range(B):
            best_gt_idx[b, best_anchor_idx[b]] = torch.arange(
                N, dtype=torch.long, device=device
            )
            best_gt_iou[b, best_anchor_idx[b]] = 2  # Make sure the box is selected

        # Filter out low IoU matches
        pos_mask = best_gt_iou > 0.5

        for b in range(B):
            anchs = anchors[b]
            valid_gt_boxes = gt_boxes[b][mask[b]]
            valid_gt_labels = gt_labels[b][mask[b]]

            if valid_gt_boxes.shape[0] == 0:
                continue  # Skip if no valid ground truth boxes

            encoded_labels[b, pos_mask[b]] = valid_gt_labels[
                best_gt_idx[b, pos_mask[b]]
            ]

            # Encode bounding boxes
            matched_gtb = valid_gt_boxes[best_gt_idx[b]]

            # Relative anchor offsets
            encoded[b, :, 0] = (matched_gtb[:, 0] - anchs[:, 0]) / anchs[:, 2]
            encoded[b, :, 1] = (matched_gtb[:, 1] - anchs[:, 1]) / anchs[:, 3]
            encoded[b, :, 2] = ((matched_gtb[:, 2] + eps) / (anchs[:, 2] + eps)).log()
            encoded[b, :, 3] = ((matched_gtb[:, 3] + eps) / (anchs[:, 3] + eps)).log()

        if not is_batched:
            # Return to original tensor shapes
            encoded = encoded.squeeze(0)
            encoded_labels = encoded_labels.squeeze(0)
            pos_mask = pos_mask.squeeze(0)

        return AnchorEncodeOutput(encoded, encoded_labels, pos_mask)

    def decode(self, encoded: torch.Tensor):
        device = encoded.device
        anchors = self.anchors.to(device)

        decoded = torch.zeros_like(encoded, device=device)

        decoded[..., 0] = decoded[..., 0] * anchors[..., 2] + anchors[..., 0]
        decoded[..., 1] = decoded[..., 1] * anchors[..., 3] + anchors[..., 1]
        decoded[..., 2] = decoded[..., 2].exp() * anchors[..., 2]
        decoded[..., 3] = decoded[..., 3].exp() * anchors[..., 3]

        return decoded
