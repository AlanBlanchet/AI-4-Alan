import multiprocessing as mp
from typing import ClassVar, Self

import lightning.pytorch.loops.evaluation_loop as pl_eval_loop
import torch
from lightning.pytorch.loops.fetchers import _PrefetchDataFetcher
from torch.utils.data import get_worker_info
from torchvision.utils import save_image

from ...dataset.env.environment import EnvironmentDatasets
from ...dataset.env.queues import RLQueues
from ...train.datamodule import DataModule
from ...utils.pydantic_ import validator
from ..metrics import Metrics
from ..task import Task
from .metrics import RLMetric


class ReinforcementLearning(Task):
    alias = "rl"
    _queues: ClassVar[RLQueues] = None
    """Queues for communication between the environment and the agent"""

    datasets: EnvironmentDatasets
    """For RL, we specifically use an EnvironmentDataset"""
    metrics: Metrics = Metrics(
        metric=lambda: RLMetric(["reward", "memory", "epsilon"]),
        groups=["train", "val"],
    )
    """RL tasks have their own metrics"""

    @validator("datamodule")
    def validate_datamodule(cls, value: dict, values):
        if "train_shuffle" not in value or value["train_shuffle"]:
            cls.warn("Explicitely setting train_shuffle to False for RL")
            value["train_shuffle"] = False
        # We need to create the queues before the datamodule is created

        ctx = mp.get_context("fork")
        cls._queues = RLQueues(ctx=ctx, num_workers=value["num_workers"])
        return DataModule(
            dataset=values["dataset"], **{**value, **cls.datamodule_kwargs()}
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
        dataset.init_worker(worker_id, split_queue)

    def init(self):
        super().init()
        # Patch the data fetcher in the EvalLoop to use our custom one
        # This prevents double fetching at the start which we don't want

        def _patched_select_data_fetcher(trainer, stage):
            return RLPrefetchDataFetcher()

        pl_eval_loop._select_data_fetcher = _patched_select_data_fetcher

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


class RLPrefetchDataFetcher(_PrefetchDataFetcher):
    """Custom data fetcher for RL tasks"""

    def __init__(self, prefetch_batches=1):
        super().__init__(prefetch_batches)
        assert prefetch_batches == 1, "RL tasks only support prefetch_batches=1"

    def __iter__(self) -> Self:
        """Removes first fetching of data in the prefetch iterator"""
        self.iterator = iter(self.combined_loader)
        self.reset()
        return self
