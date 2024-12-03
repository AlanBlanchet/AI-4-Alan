# from __future__ import annotations

# from typing import TYPE_CHECKING, Any

# import plotly.express as px
# import torch
# import torch.optim as optim
# from lightning import LightningModule
# from torch.optim.lr_scheduler import ExponentialLR

# if TYPE_CHECKING:
#     from ..task.task import Task


# class AIModule(LightningModule):
#     def __init__(self, task: Task):
#         super().__init__()

#         self.task = task
#         self.model = task.model
#         self.log_ = None

#         if hasattr(task, "logger"):
#             self.log_ = task.logger.log

#         # INFO: for lighting device handling
#         for k, module in self.task.modules().items():
#             setattr(self, k, module)

#         self.set_random_example_idx()

#     @property
#     def config(self):
#         return self.task.root_config

#     @property
#     def datamodule_config(self):
#         return self.config.run.datamodule.config

#     def model_parameters(self):
#         return self.model.parameters()

#     def log(self, name: str, value: Any, prog_bar: bool = False):
#         # Override for custom logger
#         if isinstance(value, torch.Tensor) and (value.ndim == 0 or value.numel() == 1):
#             super().log(name, value, prog_bar=prog_bar, sync_dist=True)
#         elif self.log_ is not None:
#             self.log_(name, value, prog_bar=prog_bar)

#     def forward(self, x):
#         if isinstance(x, list):
#             return self.model(*x)
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         return self.step(batch, batch_idx, "train", plot_idx=self.random_train_idx)

#     def validation_step(self, batch, batch_idx):
#         return self.step(batch, batch_idx, "val", plot_idx=self.random_val_idx)

#     def step(self, batch, batch_idx: int, split: str, plot_idx: list[int] = None):
#         # Get the possible item to show as example
#         example_item_idx = (
#             None if (plot_idx is None or plot_idx[0] != batch_idx) else plot_idx[1]
#         )
#         # Process the batch
#         losses = self.task.process(
#             model=self, batch=batch, split=split, item_idx=example_item_idx
#         )
#         # Log the losses
#         for k, v in losses.items():
#             # Change the precision of the loss to be a fixed 6
#             v /= self.datamodule_config.batch_size
#             self.log(
#                 f"{split}/{k}",
#                 v,
#                 prog_bar=split == "train",
#             )

#         # INFO : This is the only way to get the worker info
#         # torch.utils.data.get_worker_info()

#         # Return the losses
#         return losses

#     def log_metric(self, metric: dict, split: str):
#         for k, v in metric.items():
#             if isinstance(v, dict):
#                 v = {f"{k}/{kk}": vv for kk, vv in v.items()}
#                 self.log_metric(v, split)
#             elif isinstance(v, torch.Tensor):
#                 if v.ndim < 2:
#                     if v.ndim == 1:
#                         v = v.mean()

#                     if self.log_ is not None:
#                         self.log_(
#                             k,
#                             v,
#                             logger=True,
#                             prog_bar="training" in k and split == "train",
#                         )
#                     else:
#                         self.log(k, v, prog_bar="training" in k and split == "train")
#                 elif v.ndim == 2:
#                     if v.shape[0] == v.shape[1]:
#                         # Supposed to be a confusion matrix
#                         fig = px.imshow(
#                             v.cpu(),
#                             labels=dict(
#                                 x="Prediction", y="Ground Truth", color="Score"
#                             ),
#                             x=self.task.label_map.labels,
#                             y=self.task.label_map.labels,
#                         )
#                         fig.layout.height = v.shape[-1] * 50
#                         fig.layout.width = v.shape[-1] * 50
#                         if self.log_ is not None:
#                             self.log_(f"{k}", fig)

#     def epoch_end(self, split: str):
#         metrics = self.metrics.compute(split=split)
#         metrics = {f"{split}/{k}": v for k, v in metrics.items()}
#         self.log_metric(metrics, split)

#     def on_train_epoch_start(self) -> None:
#         self.train()

#     def on_train_epoch_end(self):
#         if not self.task.val_only_metrics:
#             self.epoch_end("train")

#     def on_validation_epoch_end(self):
#         self.set_random_example_idx()
#         self.epoch_end("val")

#     def configure_optimizers(self):
#         opt = optim.AdamW(self.model.parameters(), lr=4e-4)
#         return dict(
#             optimizer=opt,
#             lr_scheduler=dict(
#                 scheduler=ExponentialLR(optimizer=opt, gamma=0.995),
#                 interval="epoch",
#             ),
#         )

#     def get_random_example_idx(self, dataset):
#         if hasattr(dataset, "__len__"):
#             B = self.datamodule_config.batch_size
#             return (
#                 torch.randint(0, len(dataset) // B - 1, (1,)).item(),
#                 torch.randint(0, B, (1,)).item(),
#             )
#         return None

#     def set_random_example_idx(self):
#         if self.task.train:
#             self.random_train_idx = self.get_random_example_idx(
#                 self.task.dataset.get_train()
#             )

#         self.random_val_idx = self.get_random_example_idx(self.task.dataset.get_val())

#     @property
#     def metric_val_only(self):
#         return self.task.config.task.metric_val_only
