from typing import Any, Generator, Mapping, TypeVar

import torch
import torch.nn as nn
from einops import rearrange

from ..env.environment import Environment
from ..utils.func import keep_kwargs_prefixed
from ..utils.hyperparam import HYPERPARAM, parse_hyperparam

T = TypeVar("T")


class Agent(nn.Module):
    def __init__(
        self,
        env: Environment,
        policy: nn.Module,
        epsilon: HYPERPARAM = 0,
        history=0,
        requires_merge=True,
        **kwargs,
    ):
        super().__init__()
        self.states = []

        self.policy = policy
        self.drop(env)

        self.lifetime = 0
        self.history = history
        self.requires_merge = requires_merge
        self.epsilon = parse_hyperparam(
            epsilon, end=0.1, **keep_kwargs_prefixed(kwargs, "epsilon_")
        )

        self.register_state(["lifetime", "history", "requires_merge"])

    @property
    def memory(self):
        return self._env.memory

    def prepare(self) -> Generator[Any, Any, Any]: ...

    def interact(self):
        delayed = list(self.memory.delayed_key_map.keys())
        # Always add a history dimension
        obs, *delayed_obs = self.last(self.history)[delayed]
        obs = self._env.preprocess(obs)
        action, *storables = self.act(obs, *delayed_obs)
        # Collect experience
        experience = self._env.step(action, *storables)
        self.lifetime += 1
        return experience

    @property
    def view(self):
        return self._env.view

    def episode_interact(self, n: int):
        dones = 0
        while dones < n:
            data = self.interact()[:5]
            *_, done = data
            dones += done
            yield data

    def forward(self, obs):
        return self.act(obs)

    def act(self, obs: torch.Tensor, *others: torch.Tensor):
        other = ()
        if self.epsilon != 0 and torch.rand(1) < self.epsilon:
            t = torch.rand(self._env.out_action)
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
            trajectory["obs"] = self._env.preprocess(trajectory["obs"])
            trajectory["next_obs"] = self._env.preprocess(trajectory["next_obs"])
            yield trajectory.to(self.device)

    def sample(self, size: int, history: int = 0, priority: str = None):
        samples = self.memory.sample(size, history, priority=priority)
        samples["obs"] = self._env.preprocess(samples["obs"])
        samples["next_obs"] = self._env.preprocess(samples["next_obs"])
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
        self._env.setup_delayed(delayed_key_map, shapes)

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
        self._env = env

    def step(self, train_step: int):
        self.epsilon.step(train_step)

    def learn(self) -> dict: ...

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
        self.epsilon.train()
        return super().train(mode)

    def eval(self, env: Environment):
        super().eval()
        self.epsilon.eval()
        self.drop(env)
