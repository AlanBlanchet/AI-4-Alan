import multiprocessing as mp
from multiprocessing import Queue
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import field_validator
from torch.utils.data import get_worker_info
from torchvision.utils import save_image

from ...dataset.env.environment import EnvironmentDataset
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

    _agent2env_queues: ClassVar[list[Queue]] = None
    _env2agent_queues: ClassVar[list[Queue]] = None

    @field_validator("datamodule", mode="before")
    @classmethod
    def validate_datamodule(cls, v: dict, others):
        if "train_shuffle" not in v or v["train_shuffle"]:
            cls.warn("Explicitely setting train_shuffle to False for RL")
            v["train_shuffle"] = False
        # We need to create the queues before the datamodule is created
        cls.create_queues(v["num_workers"])
        return AIDataModule(
            dataset=others.data["dataset"], **{**v, **cls.datamodule_kwargs()}
        )

    @classmethod
    def create_queues(cls, workers: int):
        if cls._agent2env_queues is None and cls._env2agent_queues is None:
            cls.info(f"Creating {workers} queues (actions, states) for RL")
            ctx = mp.get_context("fork")
            cls._agent2env_queues = [ctx.Queue() for _ in range(workers)]
            cls._env2agent_queues = [ctx.Queue() for _ in range(workers)]

    @classmethod
    def model_kwargs(cls, params):
        # We make the env communicate with the agent through queues
        import os

        # Disable CUDA for this worker
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        return dict(
            env=params["dataset"],
            agent2env=cls._agent2env_queues,
            env2agent=cls._env2agent_queues,
        )

    @classmethod
    def datamodule_kwargs(cls):
        return dict(worker_init_fn=cls._worker_init_fn, val_batch_size=1)

    @classmethod
    def _worker_init_fn(cls, worker_id: int):
        worker_info = get_worker_info()
        if not worker_info:
            raise RuntimeError("worker_init_fn must be called in a worker process")

        dataset = worker_info.dataset  # Access the dataset directly
        dataset.init_worker(
            worker_id,
            agent2env=cls._agent2env_queues[worker_id],
            env2agent=cls._env2agent_queues[worker_id],
        )

    def default_loss(self, out: dict, batch: dict) -> dict:
        raise NotImplementedError

    def setup_dataset(self, **kwargs):
        self.dataset.prepare(
            run_p=self.run_p,
            batch_size=self.datamodule.batch_size,
            val_batch_size=self.datamodule.val_batch_size,
        )

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

    def process_output(
        self, model: nn.Module, batch: dict[str, torch.Tensor], split: str
    ) -> dict:
        input = batch["input"]
        labels = batch["labels"]

        out: torch.Tensor = model(input)

        loss = F.cross_entropy(out, labels)

        self.metrics.update(out, labels, split=split)

        return dict(loss=loss)
