from __future__ import annotations

from typing import TYPE_CHECKING, Any

import plotly.express as px
import torch
import torch.optim as optim
from lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR

from ..registry import REGISTER
from .defaults import BATCH_SIZE

if TYPE_CHECKING:
    from ..task.task import Task


@REGISTER
class AIModule(LightningModule):
    def __init__(self, task: Task):
        super().__init__()

        self.task = task
        self.model = task.model
        self.log_ = None

        if hasattr(task, "logger"):
            self.log_ = task.logger.log

        # INFO: for lighting device handling
        for k, module in self.task.modules().items():
            setattr(self, k, module)

        # def init_weights(m):
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         m.bias.data.fill_(0.01)
        #     elif isinstance(m, nn.Conv2d):
        #         torch.nn.init.xavier_uniform_(m.weight)

        # self.model.apply(init_weights)

        self.random_batch_idx = None

    def log(self, name: str, value: Any, prog_bar: bool = False):
        # Override for custom logger
        if isinstance(value, torch.Tensor) and (value.ndim == 0 or value.numel() == 1):
            super().log(name, value, prog_bar=prog_bar)
        elif self.log_ is not None:
            self.log_(name, value, prog_bar=prog_bar)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val", plot_idx=self.random_batch_idx)

    def step(self, batch, batch_idx: int, split: str, plot_idx: list[int] = None):
        losses = self.task.process(self.model, batch, split)
        for k, v in losses.items():
            self.log(f"{split}/{k}", v, prog_bar=split == "train")
        return losses

        # for b in range(len(decoded_locs)):
        #     b_max_labels = max_labels[b]
        #     score_mask = b_max_labels > 0.7
        #     filtered_max_labels = b_max_labels[score_mask]
        #     filtered_max_labels_idx = max_labels_idx[b][score_mask]
        #     filtered_locs_xy = decoded_locs_xy[b][score_mask]

        #     kept_idx = batched_nms(
        #         filtered_locs_xy, filtered_max_labels, filtered_max_labels_idx, 0.5
        #     )
        #     kept_locs = filtered_locs_xy[kept_idx]
        #     kept_scores = filtered_max_labels[kept_idx]
        #     kept_ids = filtered_max_labels_idx[kept_idx]
        #     # Remove backgrounds
        #     bg_mask = kept_ids != 0
        #     kept_locs = kept_locs[bg_mask]
        #     kept_ids = kept_ids[bg_mask]

        #     masked_gt_locs = gt_locs[b][mask[b]]
        #     masked_gt_ids = label_ids[b][mask[b]]

        #     # Example
        #     if not is_train and plot_idx is not None and (batch_idx, b) == plot_idx:
        #         ex_size = 800
        #         ex_image = image[b] * 255  # (C, H, W)
        #         ex_kept_locs = (kept_locs * ex_size).cpu()  # Rescale
        #         ex_gt_locs = masked_gt_locs.cpu() * ex_size
        #         ex_kept_labels = [
        #             f"{label} ( {score*100:.2f}% )"
        #             for label, score in zip(
        #                 self.dataset.label_map[kept_ids], kept_scores
        #             )
        #         ]
        #         ex_gt_ids = masked_gt_ids.cpu()
        #         ex_gt_labels = self.dataset.label_map[ex_gt_ids]

        #         # Gt boxes
        #         ex_image = (
        #             resize(ex_image, (ex_size, ex_size), antialias=True)
        #             .cpu()
        #             .to(dtype=torch.uint8)
        #         )
        #         ex_image = draw_bounding_boxes(
        #             image=ex_image,
        #             boxes=ex_gt_locs,
        #             colors=[(0, 0, 255)] * ex_gt_locs.size(0),
        #             labels=ex_gt_labels,
        #         )
        #         # Pred boxes
        #         ex_image = draw_bounding_boxes(
        #             image=ex_image,
        #             boxes=ex_kept_locs,
        #             colors=[(255, 0, 0)] * kept_locs.shape[0],
        #             labels=ex_kept_labels,
        #         )

        #         self.log(f"{split}/nms_boxes", kept_ids.size(0))
        #         self.logger.experiment.add_image(
        #             f"image/{split}", ex_image.numpy(), self.current_epoch
        #         )

        # return loss

    def log_metric(self, metric: dict, split: str):
        for k, v in metric.items():
            if isinstance(v, dict):
                v = {f"{k}/{kk}": vv for kk, vv in v.items()}
                self.log_metric(v, split)
            elif isinstance(v, torch.Tensor):
                if v.ndim < 2:
                    if v.ndim == 1:
                        v = v.mean()

                    if self.log_ is not None:
                        self.log_(
                            k,
                            v,
                            logger=True,
                            prog_bar="training" in k and split == "train",
                        )
                    else:
                        self.log(k, v, prog_bar="training" in k and split == "train")
                elif v.ndim == 2:
                    if v.shape[0] == v.shape[1]:
                        # Supposed to be a confusion matrix
                        fig = px.imshow(
                            v.cpu(),
                            labels=dict(
                                x="Prediction", y="Ground Truth", color="Score"
                            ),
                            x=self.task.label_map.labels,
                            y=self.task.label_map.labels,
                        )
                        fig.layout.height = v.shape[-1] * 50
                        fig.layout.width = v.shape[-1] * 50
                        if self.log_ is not None:
                            self.log_(f"{k}", fig)

    def epoch_step(self, split: str):
        metrics = self.metrics.compute(split=split)
        metrics = {f"{split}/{k}": v for k, v in metrics.items()}
        self.log_metric(metrics, split)

    def on_train_epoch_end(self):
        if not self.metric_val_only:
            self.epoch_step("train")

    def on_validation_epoch_end(self):
        self.random_batch_idx = (
            torch.randint(0, len(self.task.dataset.val()) // BATCH_SIZE, (1,)).item(),
            torch.randint(0, BATCH_SIZE, (1,)).item(),
        )

        self.epoch_step("val")

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=4e-4)
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=ExponentialLR(optimizer=opt, gamma=0.995),
                interval="epoch",
            ),
        )

    @property
    def metric_val_only(self):
        return self.task.config.task.metric_val_only
