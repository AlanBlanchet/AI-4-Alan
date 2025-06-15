from __future__ import annotations

import random
import shutil
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, override

import numpy as np
import plotly.express as px
import torch
import torch.optim as optim
import yaml
from lightning import LightningModule
from lightning.pytorch.core.saving import _default_map_location
from lightning.pytorch.utilities.migration import pl_legacy_patch
from lightning.pytorch.utilities.migration.utils import _pl_migrate_checkpoint
from pydantic import computed_field, field_validator
from pydantic.fields import FieldInfo
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from ..configs.log import Color
from ..dataset.dataset import Datasets
from ..nn import *  # noqa
from ..nn.compat.module import Module, ModuleConfig
from ..nn.compat.pretrained import Pretrained, PretrainedConfig
from ..train.datamodule import DataModule
from ..train.optimizers import Optimizer
from ..utils.env import AIEnv
from ..utils.func import create_path
from .metrics import EmptyMetric, Metrics

TASK_TYPE = Literal["binary", "multiclass", "multilabel"]


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


class LinkInfo(FieldInfo):
    links: dict[str, str]


def Link(links: dict[str, str] = {}, **kwargs):
    return LinkInfo.from_field(**kwargs)


class Task(Module, LightningModule, buildable=False):
    XT_ADD_KEYS: ClassVar[list[str]] = [
        "_log_hyperparams",
        "prepare_data_per_node",
        "allow_zero_length_dataloader_with_multiple_devices",
        "parameters",
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
        "_hparams",
    ]
    XT_REMOVE_KEYS: ClassVar[list[str]] = ["forward"]

    log_name: ClassVar[str] = "task"
    color: ClassVar[str] = Color.magenta
    alias: ClassVar[Optional[str]] = None

    metrics: Metrics = Metrics(
        metric=EmptyMetric,
        groups=["train", "val"],
    )
    """The metrics used"""
    checkpoint: Path = None
    """The checkpoint to load"""
    val_only_metrics: bool = False
    """If the metrics should only be computed on the validation set"""
    datasets: Datasets
    """The dataset used. Is required to be defined before model"""
    datamodule: DataModule
    """The datamodule used."""
    model: Module
    """The model used. Is required to be defined after dataset"""
    optimizer: Optimizer

    @classmethod
    def configure(cls, config):
        config["datasets"] = Datasets(datasets=config.pop("datasets"))
        config["datamodule"] = DataModule(
            datasets=config["datasets"], **config["datamodule"]
        )
        config["model"] = Module.from_config(config["model"] | cls.model_kwargs(config))
        config["optimizer"] = {"params": config["model"].parameters()} | config.get(
            "optimizer", dict()
        )
        return super().configure(config)

    @classmethod
    def load_from_checkpoint(cls, **kwargs):
        """Remove the original method for loading checkpoints since it doesn't allow loading directly from instance"""
        raise NotImplementedError("Use the `load` method instead")

    @classmethod
    def log_extras(cls):
        return ""

    @override
    @classmethod
    def get_identifiers(cls):
        alias = {cls.alias} if cls.alias else set()
        return super().get_identifiers() | alias

    @classmethod
    def all_names(self):
        return [task.name for task in self.all()]

    @classmethod
    def get_all_valid_tasks(cls, field: str, dtype: str):
        for task in cls.all():
            if task.is_valid(field, dtype):
                yield task

    @classmethod
    def model_kwargs(cls, params: dict):
        return {}

    @classmethod
    def datamodule_kwargs(cls, params: dict):
        return {}

    @computed_field()
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def config(self):
        return self.root_config

    # # TODO: fix this to be on each subclass
    # @property
    # def required_fields(self):
    #     if self.type == "classification":
    #         return ["labels"]
    #     elif self.type == "detection":
    #         return ["labels", "bbox"]
    #     else:
    #         raise ValueError(f"Unknown task type {self.type}")

    @cached_property
    def run_p(self):
        run_p = (
            AIEnv.runs_p / self.alias / self.model.__class__.__name__
            # / self.datasets.name.replace("/", "_")
        )
        run_p = create_path(run_p)
        run_p.mkdir(exist_ok=True, parents=True)
        self.info(f"Setting run path to {run_p}")
        return run_p

    def setup_dataset(self, **kwargs): ...

    # @abstractmethod
    # def default_loss(self, out: dict, batch: dict) -> dict: ...

    # @abstractmethod
    # def example(self, pred: dict, item: dict, split: str): ...

    def init(self):
        super().init()

        self.log_ = None

        # if hasattr(task, "logger"):
        #     self.log_ = task.logger.log

        self.set_random_example_idx()

        # Move the logs to the run path
        shutil.move(AIEnv.tmp_log_p, self.run_p / "log")
        # Load the dataset
        self.info(f"Loading {self.spaced_name()} task")
        self.setup_dataset()

    def setup(self, stage):
        if isinstance(self.model, Pretrained):
            ex = self.datasets.example
            items = masked_collator([ex])
            input = self.datasets.extract_inputs(items)
            self.model.init_weights(*input)

        return super().setup(stage)

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
        _, losses = self.process(
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
        """Logs the metrics to the logger"""
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

    def prepare_input(self, batch: dict) -> dict:
        """Called before feeding the batch to the model"""
        return self.datasets.prepare_for_model(batch)

    def process(
        self,
        model: LightningModule,
        batch: dict[str, torch.Tensor],
        split: str,
        item_idx: int | None,
    ) -> dict:
        """Main process function for the task"""
        # Extract model required inputs
        inputs = self.prepare_input(deepcopy(batch))

        # Forward pass
        out = model(inputs)

        # Format as dict
        out = self.wrap_output(out)

        # Compute the loss if needed
        if "loss" in out:
            losses = {k: v for k, v in out.items() if "loss" in k}
        elif hasattr(self.model, "compute_loss"):
            losses = self.model.compute_loss(out, batch)
        elif hasattr(self, "default_loss"):
            losses = self.default_loss(out, batch)
        else:
            losses = {}  # No loss, only metrics / predictions

        # Format losses
        if isinstance(losses, torch.Tensor):
            losses = dict(loss=losses)
        elif isinstance(losses, (list, tuple)):
            losses = dict(loss=losses[0], other=losses[1:])

        # Postprocess the output
        if hasattr(self.model, "postprocess"):
            out = self.postprocess(out, batch)

        # Update metrics
        self.metrics.update(out, batch, split=split)

        # Show chosen sample
        if item_idx is not None:
            self.example(
                self._extract_batch_item(out, item_idx),
                self._extract_batch_item(batch, item_idx),
                split,
            )

        # Return the losses
        return out, losses

    def wrap_output(self, out: dict):
        return out

    def map_params(self, item: dict) -> dict:
        dataset_conf = self.root_config.dataset
        map = dataset_conf.map_params
        return self.datasets.parse_items(item, map)

    def _extract_batch_item(self, batch: dict, idx: int) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, (list, torch.Tensor, np.ndarray)):
                out[k] = v[idx]
            else:
                out[k] = v
        return out

    def save_config(self):
        config = self.run_p / "config.yml"
        dump_config = self.run_p / "dump.yml"
        config.write_text(yaml.dump(self.root_config.config, sort_keys=False))

        # print(self.hparams)
        # exit(0)
        # obj = self.root_config.model_dump(exclude={"config"})
        # try:
        #     dump_config.write_text(yaml.dump(obj, sort_keys=False))
        # except Exception as e:
        #     self.error(f"Error saving dump config:\n{e}\nTrying to save as json")
        #     try:
        #         dump_config.with_suffix(".json").write_text(json.dumps(obj, indent=2))
        #     except Exception as ee:
        #         raise ee from e

        self.info(f"Saved configs to \nconfig {config}\ndump {dump_config}")

    def get_random_example_idx(self, dataloader: DataLoader):
        """Returns random example index for the dataset"""
        if hasattr(dataloader, "__len__"):
            B = self.datamodule.batch_size
            return (random.randint(0, len(dataloader)), random.randint(0, B))
        return None

    def set_random_example_idx(self):
        """Sets the random example index for the training and validation set"""
        if self.training:
            self.random_train_idx = self.get_random_example_idx(
                self.datamodule.train_dataloader()
            )

        self.random_val_idx = self.get_random_example_idx(
            self.datamodule.val_dataloader()
        )

    def load(self, checkpoint: Path, map_location=None):
        """Load the checkpoint into the model"""
        map_location = map_location or _default_map_location
        with pl_legacy_patch():
            ckpt = torch.load(checkpoint, map_location=map_location)

        # convert legacy checkpoints to the new format
        ckpt = _pl_migrate_checkpoint(ckpt, checkpoint_path=checkpoint)

        self.warn("Only loading the state_dict. TODO: Load the full checkpoint")
        self.load_state_dict(ckpt["state_dict"])

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


class DimensionConfig(ModuleConfig):
    num_channels: int
    fixed_size: list[int] = None

    @field_validator("fixed_size", mode="before")
    @classmethod
    def validate_fixed_size(cls, value):
        if isinstance(value, int):
            return [value, value]
        if value is None and issubclass(cls, PretrainedConfig):
            return cls.pretrained[0].weights.fixed_size
        return value


class ClassificationConfig(ModuleConfig):
    num_classes: int = None

    @field_validator("num_classes", mode="before")
    @classmethod
    def validate_num_classes(cls, value):
        if value is None and issubclass(cls, PretrainedConfig):
            return cls.pretrained_recommendations[0].weights.num_classes
        return value
