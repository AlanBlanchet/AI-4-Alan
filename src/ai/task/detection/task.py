from functools import cached_property
from typing import ClassVar

from ...modality.image import Image
from ..classification.task import Classification
from ..metrics import GroupedMetric
from .metrics import DetectionMetrics


class Detection(Classification):
    name: ClassVar[str] = "detection"
    alias: ClassVar[str] = "det"

    @cached_property
    def metrics(self):
        return GroupedMetric(
            lambda: DetectionMetrics(num_classes=len(self.label_map)),
            ["train", "val"],
        )

    def threshold(self, out, threshold=0.3):
        score, bbox, labels = out["scores"], out["bbox"], out["labels"]
        mask = score > threshold
        return dict(
            scores=score[mask],
            bbox=bbox[mask],
            labels=labels[mask],
        )

    def example(self, out: dict, item: dict, split: str):
        # Threshold the output to get most important predictions
        pred = self.threshold(out)

        # Postprocess the input to retrieve the original image
        item = self.dataset.image_modality.postprocess(item, split)

        # Plot the image
        Image.plot(
            path=self.run_p / f"{item["id"].item()}_{split}.png",
            image=item["image"],
            bbox=pred["bbox"],
            labels=self.label_map[pred["labels"]],
            scores=pred["scores"],
            gt_bbox=item["bbox"],
            gt_labels=self.label_map[item["labels"]],
        )

    def postprocess(self, out: dict, batch) -> dict:
        # Postprocess the output
        return self.model.postprocess(out, batch)
