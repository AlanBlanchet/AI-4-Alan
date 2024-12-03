from abc import abstractmethod
from queue import Empty
from typing import Any, Literal, Mapping, TypeVar

import torch
import torch.nn as nn
from pydantic import field_validator

from ...dataset.env.environment import Environment, EnvironmentDataset
from ...dataset.env.state import StateDict
from ...utils.hyperparam import HYPERPARAM, Hyperparam
from .module import Module

T = TypeVar("T")


class Agent(Module, buildable=False):
    env: EnvironmentDataset
    optimizer: Literal["Adam", "RMSprop", "AdamW"] = "RMSProp"
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon: HYPERPARAM = Hyperparam(start=0.9, end=0.1)
    batch_size: int = 32
    history: int = 4
    prepare_episodes: int = 4
    reward_shaping: bool = True
    interactions_per_learn: int = 1
    policy: nn.Module = None

    env2agent: list[Any]
    agent2env: list[Any]

    _env_info: dict[str, Any] = {}
    _current_env: Environment = None
    _current_state: Any = []
    _states = []
    _lifetime = 0
    _requires_merge: bool = False
    _queue_idx = 0

    @field_validator("policy", mode="before")
    def validate_policy(cls, value, values):
        if value is None:
            return None
        elif isinstance(value, nn.Module):
            return value
        elif isinstance(value, dict):
            return Module.from_config(value)

    @field_validator("history", mode="before")
    def validate_history(cls, value, values):
        history = max(value, 1)
        if len(values["env"].preprocessed_shape) <= 1:
            history = 1
        return history

    def init(self):
        super().init()

        if self.is_mp:
            self._current_state = [None for _ in range(len(self.env2agent))]
        else:
            self._current_state = None

        self._current_env = self.env.get_train()

        if self.policy is None:
            self.policy = self.setup_policy()

    @property
    def memory(self):
        return self.active_env.buffer

    @property
    def active_env(self):
        return self._current_env

    @property
    def is_mp(self):
        return len(self.env2agent) > 0

    @property
    def state(self):
        if self.is_mp:
            return self._current_state[self._queue_idx]
        return self._current_state

    @state.setter
    def state(self, value):
        if self.is_mp:
            self._current_state[self._queue_idx] = value
        else:
            self._current_state = value

    @property
    def view(self):
        return self.active_env.view

    @abstractmethod
    def setup_policy(self) -> nn.Module: ...

    # def prepare(self) -> Generator[Any, Any, Any]:
    #     for _ in range(4):
    #         self.active_env.step(action, *storables)

    def _get_state(self) -> StateDict:
        if self.is_mp:
            try:
                self.state, self._env_info = self.env2agent[self._queue_idx].get()
            except Empty as e:
                self.error("Queue timed out", e)
                raise e
        else:
            self.state = self.memory.last(self.history)
            self._env_info = dict(memory=len(self.memory))
        return self.state

    def _send_act(self, action):
        if self.is_mp:
            self.agent2env[self._queue_idx].put(action)
        else:
            self.active_env.step(action)
        self.step(self._lifetime)

    def env_info(self):
        return self._env_info

    def interact(self):
        # delayed = list(self.memory.delayed_key_map.keys())
        obs = self._get_state()["next_obs"]

        # Preprocess the observations
        obs = self.active_env.preprocess(obs)

        # FIXME putting this in the forward might not be the best idea
        # It will take time to put on the device
        obs = torch.as_tensor(obs, device=self.device)

        action, *storables = self.act(obs)
        action = action.detach().argmax().cpu().numpy()
        # Send the action to the environment
        self._send_act(action)

        self._lifetime += 1

        if self.is_mp:
            self._queue_idx += 1
            self._queue_idx %= len(self.env2agent)

    def forward(self, batch):
        [self.interact() for _ in range(self.interactions_per_learn)]
        self.step(self._lifetime)
        return self.learn(batch)

    def act(self, obs: torch.Tensor, *others: torch.Tensor):
        other = ()
        if self.epsilon != 0 and torch.rand(1) < self.epsilon:
            t = torch.rand(self.active_env.out_action)
            action = (t - t.min()) / (t.max() - t.min() + 1e-6)
            if len(others) > 0:
                elems = [obs.unsqueeze(dim=0)] + [o.unsqueeze(dim=0) for o in others]
                _, *other = self.process_act(self.policy(*elems))
        else:
            elems = [obs.unsqueeze(dim=0)] + [o.unsqueeze(dim=0) for o in others]
            action, *other = self.process_act(self.policy(*elems))
        return (action, *other)

    def process_act(self, x: torch.Tensor | tuple[torch.Tensor | Any]):
        if isinstance(x, torch.Tensor):
            x = [x]
        if isinstance(x, (list, tuple)):
            x = [y for y in x if y is not None]
        data = self.unbatch(x)
        data[0] = data[0].softmax(dim=-1)
        return data

    def unbatch(self, data: Any):
        if isinstance(data, (list, tuple)):
            return [self.unbatch(d) for d in data]
        return data.unsqueeze(dim=0)

    def last(self, history: int = 0):
        if self.is_mp:
            raise NotImplementedError
        return self.memory.last(history).to(self.device)

    def setup_delayed(
        self, delayed_key_map: dict[str, str], shapes: list[tuple[int, ...]]
    ):
        self.active_env.setup_delayed(delayed_key_map, shapes)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        state_dict.update({k: self.__getattribute__(k) for k in self._states})
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        [setattr(self, k, state_dict.pop(k)) for k in self._states if k in state_dict]
        super().load_state_dict(state_dict, strict)

    def register_state(self, names: str | list[str]):
        self._states.extend(names if isinstance(names, (list, tuple)) else [names])

    def drop(self, env: Environment):
        if not self.is_mp:
            self._current_env = env

    def step(self, train_step: int):
        if self.training:
            self.epsilon.step(train_step)

    def memorize(self, memo: dict):
        self.memory.memorize(memo)

    @property
    def device(self):
        return next(self.parameters()).device

    def optimize(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        clip=None,
    ):
        optimizer.zero_grad()
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad.clip_grad_norm_(self.parameters(), clip)
            # for group in optimizer.param_groups:
            #     for param in group["params"]:
            #         if param.grad is not None:
            #             param.grad.data.clamp_(-clip, clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    def train(self, mode: bool = True):
        if mode:
            self.epsilon.train_()
            self.drop(self.env.get_train())
        else:
            self.epsilon.eval_()
            self.drop(self.env.get_val())
        return super().train(mode)
