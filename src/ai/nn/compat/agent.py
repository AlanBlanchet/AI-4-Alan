from abc import abstractmethod
from functools import cached_property
from typing import Any, Mapping, TypeVar

import torch
from einops import rearrange

from ...dataset.env.environment import Environment
from ...utils.func import keep_kwargs_prefixed
from ...utils.hyperparam import parse_hyperparam
from ..arch.dqn.config import DQNConfig
from .module import Module

T = TypeVar("T")


class Agent(Module):
    def __init__(
        self,
        config: DQNConfig,
        # env: Environment,
        # policy: nn.Module,
        # epsilon: HYPERPARAM = 0,
        # history=0,
        # requires_merge=True,
        **kwargs,
    ):
        super().__init__(config)
        self.config: DQNConfig
        self.states = []

        self.env = None

        self.lifetime = 0
        self.history = config.history
        self.requires_merge = config.requires_merge
        self.epsilon = parse_hyperparam(
            config.epsilon, end=0.1, **keep_kwargs_prefixed(kwargs, "epsilon_")
        )

        self.train()
        self.register_state(["lifetime", "history", "requires_merge"])

        self.env.burn(self.history)

    @property
    def memory(self):
        return self.env.buffer

    @cached_property
    def policy(self):
        return self.build_policy()

    @abstractmethod
    def build_policy(self): ...

    # def prepare(self) -> Generator[Any, Any, Any]:
    #     for _ in range(4):
    #         self.env.step(action, *storables)

    def interact(self):
        delayed = list(self.memory.delayed_key_map.keys())
        # Always add a history dimension
        obs, *delayed_obs = self.last(self.history)[delayed]
        obs = self.env.preprocess(obs)
        action, *storables = self.act(obs, *delayed_obs)
        # Collect experience
        experience = self.env.step(action, *storables)
        self.lifetime += 1
        return experience

    @property
    def view(self):
        return self.env.view

    def episode_interact(self, n: int):
        dones = 0
        while dones < n:
            data = self.interact()[:5]
            *_, done = data
            dones += done
            yield data

    def forward(self, batch):
        [self.interact() for _ in range(self.config.interactions_per_learn)]
        self.step(self.lifetime)
        return self.learn(batch)

    def act(self, obs: torch.Tensor, *others: torch.Tensor):
        other = ()
        if self.epsilon != 0 and torch.rand(1) < self.epsilon:
            t = torch.rand(self.env.out_action)
            action = (t - t.min()) / (t.max() - t.min() + 1e-6)
            if len(others) > 0:
                elems = [obs.unsqueeze(dim=0)] + [o.unsqueeze(dim=0) for o in others]
                _, *other = self.process_act(self.policy(*elems))
        else:
            elems = [obs.unsqueeze(dim=0)] + [o.unsqueeze(dim=0) for o in others]
            action, *other = self.process_act(self.policy(*elems))
        return (action, *other)

    def process_act(self, x: torch.Tensor):
        return [o.squeeze(dim=0).softmax(dim=-1) for o in x if o is not None]

    def trajectories(self, size: int):
        for trajectory in self.memory.trajectories(size):
            trajectory["obs"] = self.env.preprocess(trajectory["obs"])
            trajectory["next_obs"] = self.env.preprocess(trajectory["next_obs"])
            yield trajectory.to(self.device)

    def sample(self, size: int, history: int = 0, priority: str = None):
        samples = self.memory.sample(size, history, priority=priority)
        samples["obs"] = self.env.preprocess(samples["obs"])
        samples["next_obs"] = self.env.preprocess(samples["next_obs"])
        if self.history > 0 and self.requires_merge:
            samples["obs"] = rearrange(samples["obs"], "b s c ... -> b (s c) ...")
            samples["next_obs"] = rearrange(
                samples["next_obs"], "b s c ... -> b (s c) ..."
            )
        return samples.to(self.device)

    def last(self, history: int = 0):
        return self.memory.last(history).to(self.device)

    def setup_delayed(
        self, delayed_key_map: dict[str, str], shapes: list[tuple[int, ...]]
    ):
        self.env.setup_delayed(delayed_key_map, shapes)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        state_dict.update({k: self.__getattribute__(k) for k in self.states})
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        [setattr(self, k, state_dict.pop(k)) for k in self.states if k in state_dict]
        super().load_state_dict(state_dict, strict)

    def register_state(self, names: str | list[str]):
        self.states.extend(names if isinstance(names, (list, tuple)) else [names])

    def drop(self, env: Environment):
        self.env = env

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
            self.drop(self.config.env.train())
        else:
            self.epsilon.eval_()
            self.drop(self.config.env.val())
        return super().train(mode)
