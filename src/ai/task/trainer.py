from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import plotly.express as px
import torch
import torch.optim as optim
from lightning import LightningModule
from lightning.pytorch.core.saving import _default_map_location
from lightning.pytorch.utilities.migration import pl_legacy_patch
from lightning.pytorch.utilities.migration.utils import _pl_migrate_checkpoint
from pydantic import Field, field_validator
from torch.optim.lr_scheduler import ExponentialLR

from ..dataset.base_dataset import BaseDataset
from ..dataset.collator.mask import masked_collator
from ..nn.compat.module import Module
from ..nn.compat.pretrained import Pretrained
from ..train.datamodule import AIDataModule

PL_MODULE_KEYS = dir(LightningModule)
PL_MODULE_KEYS.remove("forward")

# Define all keys used inside the init functions of the PL Module and bases
# For the nn.Module, we were able to get these from the type hints
# With LightningModule, these aren't present and created on the fly in the __init__
PL_MODULE_KEYS.extend(
    [
        "_log_hyperparams",
        "prepare_data_per_node",
        "allow_zero_length_dataloader_with_multiple_devices",
        "_dtype",
        "_device",
        "_auto_choose_log_on_epoch",
        "_example_input_array",
        "_trainer",
        "_example_input_array",
        "_automatic_optimization",
        "_strict_loading",
        "_current_fx_name",
        "_param_requires_grad_state",
        "_metric_attributes",
        "_compiler_ctx",
        "_fabric",
        "_fabric_optimizers",
        "_device_mesh",
    ]
)


class TaskModule(Module, LightningModule, buildable=False):
    INIT_CLS: ClassVar[type] = LightningModule
    BASE_SPECIAL_KEYS: ClassVar[list[str]] = PL_MODULE_KEYS

    dataset: BaseDataset
    """The dataset used for the task. Is required to be defined before model"""
    datamodule: AIDataModule = Field(None, validate_default=True)
    """The datamodule used for the task."""
    model: Module
    """The model used for the task. Is required to be defined after dataset"""

    @field_validator("datamodule", mode="before")
    @classmethod
    def validate_datamodule(cls, v, others):
        return AIDataModule(dataset=others.data["dataset"], **v)

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, config, others):
        cls.info("Setting up model")

        config = {**config, **cls.model_kwargs(others.data)}
        dataset = others.data["dataset"]
        ex = dataset.example
        items = masked_collator([ex])
        input = dataset.extract_inputs(items)

        model: Module = Module.from_config(config)

        if isinstance(model, Pretrained):
            model.init_weights(*input)

        return model

    @property
    def config(self):
        return self.root_config

    def init(self):
        super().init()

        self.log_ = None

        # if hasattr(task, "logger"):
        #     self.log_ = task.logger.log

        self.set_random_example_idx()

    def model_parameters(self):
        return self.model.parameters()

    def log(self, name: str, value: Any, prog_bar: bool = False):
        # Override for custom logger
        if isinstance(value, torch.Tensor) and (value.ndim == 0 or value.numel() == 1):
            self.INIT_CLS.log(self, name, value, prog_bar=prog_bar, sync_dist=True)
        elif self.log_ is not None:
            self.log_(name, value, prog_bar=prog_bar)

    def forward(self, x):
        if isinstance(x, list):
            return self.model(*x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train", plot_idx=self.random_train_idx)

    def validation_step(self, batch, batch_idx=None):
        return self.step(batch, batch_idx, "val", plot_idx=self.random_val_idx)

    def step(self, batch, batch_idx: int, split: str, plot_idx: list[int] = None):
        # Get the possible item to show as example
        example_item_idx = (
            None if (plot_idx is None or plot_idx[0] != batch_idx) else plot_idx[1]
        )
        # Process the batch
        losses = self.process(
            model=self, batch=batch, split=split, item_idx=example_item_idx
        )
        # Log the losses
        for k, v in losses.items():
            self.log(
                f"{split}/{k}",
                v,
                prog_bar=split == "train",
            )

        # Return the losses
        return losses

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
                            x=self.label_map.labels,
                            y=self.label_map.labels,
                        )
                        fig.layout.height = v.shape[-1] * 50
                        fig.layout.width = v.shape[-1] * 50
                        if self.log_ is not None:
                            self.log_(f"{k}", fig)

    def epoch_end(self, split: str):
        metrics = self.metrics.compute(split=split)
        metrics = {f"{split}/{k}": v for k, v in metrics.items()}
        self.log_metric(metrics, split)

    def on_train_epoch_start(self) -> None:
        self.train()

    def on_train_epoch_end(self):
        if not self.val_only_metrics:
            self.epoch_end("train")

    def on_validation_epoch_end(self):
        self.set_random_example_idx()
        self.epoch_end("val")

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=4e-4)
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=ExponentialLR(optimizer=opt, gamma=0.995),
                interval="epoch",
            ),
        )

    def get_random_example_idx(self, dataset):
        if hasattr(dataset, "__len__"):
            B = self.datamodule.batch_size
            return (
                torch.randint(0, len(dataset) // B - 1, (1,)).item(),
                torch.randint(0, B, (1,)).item(),
            )
        return None

    def set_random_example_idx(self):
        if self.training:
            self.random_train_idx = self.get_random_example_idx(
                self.dataset.get_train()
            )

        self.random_val_idx = self.get_random_example_idx(self.dataset.get_val())

    @classmethod
    def log_extras(cls):
        return ""

    def load(self, checkpoint: Path, map_location=None):
        map_location = map_location or _default_map_location
        with pl_legacy_patch():
            ckpt = torch.load(checkpoint, map_location=map_location)

        # convert legacy checkpoints to the new format
        ckpt = _pl_migrate_checkpoint(ckpt, checkpoint_path=checkpoint)

        self.warn("Only loading the state_dict. TODO: Load the full checkpoint")
        self.load_state_dict(ckpt["state_dict"])

    @classmethod
    def load_from_checkpoint(cls, **kwargs):
        raise NotImplementedError("Use the `load` method instead")
