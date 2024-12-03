import multiprocessing as mp
from typing import ClassVar

import lightning.pytorch.loops.evaluation_loop as pl_eval_loop
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loops.fetchers import _DataFetcher
from pydantic import field_validator
from torch.utils.data import get_worker_info
from torchvision.utils import save_image

from ...dataset.env.environment import EnvironmentDataset
from ...dataset.env.queues import RLQueues
from ...train.datamodule import AIDataModule
from ..metrics import GroupedMetric
from ..task import Task
from .metrics import RLMetric


class ReinforcementLearning(Task):
    alias: ClassVar[str] = "rl"

    dataset: EnvironmentDataset
    metrics: GroupedMetric = GroupedMetric(
        lambda: RLMetric(["reward", "memory", "epsilon"]),
        ["train", "val"],
    )

    _queues: ClassVar[RLQueues] = None

    def model_post_init(self, __context):
        # Patch the data fetcher in the EvalLoop to use our custom one
        # This prevents double fetching at the start which we don't want
        _select_data_fetcher = pl_eval_loop._select_data_fetcher

        def _patched_select_data_fetcher(trainer, stage):
            return _DataFetcher()

        pl_eval_loop._select_data_fetcher = _patched_select_data_fetcher

    @field_validator("datamodule", mode="before")
    @classmethod
    def validate_datamodule(cls, v: dict, others):
        if "train_shuffle" not in v or v["train_shuffle"]:
            cls.warn("Explicitely setting train_shuffle to False for RL")
            v["train_shuffle"] = False
        # We need to create the queues before the datamodule is created

        ctx = mp.get_context("spawn")
        cls._queues = RLQueues(ctx=ctx, num_workers=v["num_workers"])
        return AIDataModule(
            dataset=others.data["dataset"], **{**v, **cls.datamodule_kwargs()}
        )

    @classmethod
    def model_kwargs(cls, params):
        # We make the env communicate with the agent through queues
        return dict(env=params["dataset"], queues=cls._queues)

    @classmethod
    def datamodule_kwargs(cls):
        return dict(
            worker_init_fn=cls._worker_init_fn, val_batch_size=1, val_prefetch_factor=1
        )

    @classmethod
    def _worker_init_fn(cls, worker_id: int):
        worker_info = get_worker_info()
        if not worker_info:
            raise RuntimeError("worker_init_fn must be called in a worker process")

        dataset = worker_info.dataset  # Access the dataset directly
        split_queue = (
            cls._queues.val_test[worker_id]
            if dataset.is_val
            else cls._queues.train[worker_id]
        )
        cls.info("Using", id(split_queue.agent2env), id(split_queue.env2agent), "queue")
        dataset.init_worker(worker_id, split_queue)

    def default_loss(self, out: dict, batch: dict) -> dict:
        raise NotImplementedError

    def setup_dataset(self, **kwargs):
        self.dataset.prepare(run_p=self.run_p)

    def wrap_output(self, out: dict | torch.Tensor):
        if isinstance(out, torch.Tensor):
            return dict(logits=out)
        return out

    def example(self, out: dict, item: dict, split: str):
        obs = item["obs"]
        epoch = self.current_epoch

        out_p = self.run_p / f"examples/{epoch}"
        out_p.mkdir(exist_ok=True, parents=True)

        for i, channel in enumerate(obs):
            save_image(channel, out_p / f"channel_{i}.png")

    # def validation_step(
    #     self, batch, batch_idx=None, dataloader_idx=0, dataloader_iter=0
    # ):
    #     """Reinforcement learning doesn't uses iterable datasets, thus there is no batch id"""
    #     return super().validation_step(batch, batch_idx)

    def process_output(
        self, model: nn.Module, batch: dict[str, torch.Tensor], split: str
    ) -> dict:
        input = batch["input"]
        labels = batch["labels"]

        out: torch.Tensor = model(input)

        loss = F.cross_entropy(out, labels)

        self.metrics.update(out, labels, split=split)

        return dict(loss=loss)
