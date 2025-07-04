from multiprocessing import Queue
from multiprocessing.context import ForkContext
from typing import Any

from pydantic import BaseModel, PrivateAttr


class SplitQueue(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    ctx: ForkContext | Any
    _env2agent: Queue = PrivateAttr()
    _agent2env: Queue = PrivateAttr()

    @property
    def env2agent(self):
        return self._env2agent

    @property
    def agent2env(self):
        return self._agent2env

    def model_post_init(self, _):
        self._env2agent = self.ctx.Queue()
        self._agent2env = self.ctx.Queue()


class RLQueues(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    ctx: ForkContext | Any
    num_workers: int

    _train: list[SplitQueue] = PrivateAttr()
    _val_test: list[SplitQueue] = PrivateAttr()

    @property
    def train(self):
        return self._train

    @property
    def val_test(self):
        return self._val_test

    def model_post_init(self, _):
        self._train = [SplitQueue(ctx=self.ctx) for _ in range(self.num_workers)]
        self._val_test = [SplitQueue(ctx=self.ctx) for _ in range(self.num_workers)]

    def __len__(self):
        return self.num_workers
