from functools import cached_property
from typing import ClassVar

import torch
import torch.nn as nn
from albumentations import BboxParams

from ...registry.registers import SOURCE
from ..classification.task import Classification
from ..metrics import GroupedMetric
from .anchor import AnchorManager
from .losses import MultiBoxDetectionLoss
from .metrics import DetectionMetrics


@SOURCE.register
class Detection(Classification):
    name: ClassVar[str] = "detection"
    alias: ClassVar[str] = "det"

    boxes: str = "boxes"

    @cached_property
    def metrics(self):
        return GroupedMetric(
            lambda: DetectionMetrics(num_classes=len(self.label_map)),
            ["train", "val"],
        )

    def modules(self, **kwargs):
        return super().modules(anchor_manager=self.anchor_manager, **kwargs)

    def setup_dataset(self, **kwargs):
        super().setup_dataset(boxes=self.boxes, **kwargs)

    @cached_property
    def anchor_manager(self):
        anchor_params = self.params.get("anchors", {})
        return AnchorManager(
            **anchor_params, background_id=self.label_map["background"]
        )

    @cached_property
    def loss(self):
        self.log("Using MultiBoxDetectionLoss")
        return MultiBoxDetectionLoss()

    def get_transforms(self, val=False, **compose_params):
        return super().get_transforms(
            val,
            bbox_params=BboxParams(format="pascal_voc", label_fields=["labels"]),
            **compose_params,
        )

    def process(
        self, model: nn.Module, batch: dict[str, torch.Tensor], split: str
    ) -> dict:
        input = batch["input"]
        B = input.shape[0]

        gt_labels = batch["labels"]
        labels_mask = batch.get("labels_mask")
        if labels_mask is not None:
            gt_mask = batch["labels_mask"]
            gt_labels[~gt_mask] = self.label_map["background"]  # Set to be background

        gt_boxes = batch["boxes"]

        # Matching
        encoded = self.anchor_manager.match(gt_labels, gt_boxes)
        gt_encoded_labels = encoded["encoded_labels"]
        gt_encoded_boxes = encoded["encoded_boxes"]
        gt_encoded_mask = gt_encoded_labels != self.label_map["background"]

        # Prediction
        out: torch.Tensor = model(input)
        boxes = out["boxes"]
        scores = out["scores"]

        losses = self.loss(
            scores=scores,
            boxes=boxes,
            gt_labels=gt_encoded_labels,
            gt_boxes=gt_encoded_boxes,
            mask=gt_encoded_mask,
        )

        # Postprocess
        decoded_boxes = self.anchor_manager.decode(boxes)
        scores = scores.softmax(-1)
        # score_mask = scores.max(-1)[0] > 0.5
        # processed_score = scores[score_mask]
        # processed_boxes = decoded_boxes[score_mask]

        if not self.metric_val_only or split == "val":
            preds, gts = [], []
            for i in range(B):
                mask = gt_encoded_mask[i]
                if mask.sum() == 0:
                    continue
                preds.append(dict(scores=scores[i][mask], boxes=decoded_boxes[i][mask]))
                gts.append(
                    dict(
                        labels=gt_encoded_labels[i][mask],
                        boxes=gt_encoded_boxes[i][mask],
                    )
                )

            self.metrics.update(preds, gts, split=split)

        return losses
