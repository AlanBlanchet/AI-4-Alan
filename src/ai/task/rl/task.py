from functools import cached_property
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision.utils import save_image

from ...dataset.env.environment import EnvironmentDataset
from ..metrics import GroupedMetric
from ..task import Task
from .metrics import RLMetric


class EmptyMetric(torchmetrics.Metric):
    def update(self, *args, **kwargs): ...

    def compute(self, **kwargs):
        return {}


class ReinforcementLearning(Task):
    alias: ClassVar[str] = "rl"

    train_shuffle: bool = False

    dataset: EnvironmentDataset

    @classmethod
    def model_kwargs(cls, params):
        return dict(env=params["dataset"])

    def default_loss(self, out: dict, batch: dict) -> dict:
        raise NotImplementedError

    @cached_property
    def metrics(self):
        return GroupedMetric(
            lambda: RLMetric(["reward", "memory", "epsilon"]),
            ["train", "val"],
        )

    def setup_dataset(self, **kwargs):
        self.dataset.prepare()

    def wrap_output(self, out: dict | torch.Tensor):
        if isinstance(out, torch.Tensor):
            return dict(logits=out)
        return out

    def example(self, out: dict, item: dict, split: str):
        """
        In RL we can't really show an example...

        TODO Maybe show the state?
        """
        obs = item["obs"]
        epoch = self.lightning_model.current_epoch

        out_p = self.run_p / f"examples/{epoch}"
        out_p.mkdir(exist_ok=True, parents=True)

        for i, channel in enumerate(obs):
            save_image(channel, out_p / f"channel_{i}.png")

    def process_output(
        self, model: nn.Module, batch: dict[str, torch.Tensor], split: str
    ) -> dict:
        input = batch["input"]
        labels = batch["labels"]

        out: torch.Tensor = model(input)

        loss = F.cross_entropy(out, labels)

        self.metrics.update(out, labels, split=split)

        return dict(loss=loss)
