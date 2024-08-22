import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiBoxDetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        scores: torch.Tensor,
        boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_boxes: torch.Tensor,
        mask: torch.Tensor,
    ):
        scores = rearrange(scores, "b a c -> b c a")

        pos_num = mask.sum(dim=-1, keepdim=True)
        N = max(1, mask.sum())

        # Localization loss
        loc_loss = F.smooth_l1_loss(boxes[mask], gt_boxes[mask], reduction="sum") / N

        # Classification loss
        c_loss = F.cross_entropy(scores, gt_labels, reduction="none")

        # Hard negative mining
        conf_neg = c_loss.clone()
        conf_neg[mask] = 0

        _, conf_idx = conf_neg.sort(dim=-1, descending=True)
        neg_num = 3 * pos_num
        neg_mask = conf_idx.sort(dim=-1)[1] < neg_num

        # Final classification loss
        # conf_loss = (c_loss * (pos_mask + neg_mask).float()).mean(-1)
        conf_loss = (c_loss[mask].sum() + c_loss[neg_mask].sum()) / N

        total_loss = loc_loss + conf_loss

        return dict(
            loss=total_loss, loc_loss=loc_loss.detach(), conf_loss=conf_loss.detach()
        )
