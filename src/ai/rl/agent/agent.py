from typing import Any, Mapping, TypeVar

import torch
import torch.nn as nn

from ..env.environment import Environment
from ..utils.hyperparam import HYPERPARAM, parse_hyperparam

T = TypeVar("T")


class Agent(nn.Module):
    def __init__(self, env: Environment, policy: nn.Module, epsilon: HYPERPARAM = 0):
        super().__init__()
        self.states = []

        self.policy = policy
        self.drop(env)

        self.lifetime = 0
        self.epsilon = parse_hyperparam(epsilon)

        self.prepare()
        self.register_state("lifetime")

    @property
    def memory(self):
        return self._env.memory

    def prepare(self):
        ...

    def interact(self):
        obs = self._env._current_obs
        obs = self._env.preprocess_fn(obs)
        action = self.act(obs)
        # Collect experience
        experience = self._env.step(action)
        self.lifetime += 1
        return experience

    def episode_interact(self, n: int):
        dones = 0
        while dones < n:
            *_, done = self.interact()
            dones += done

    def forward(self, obs):
        return self.act(obs)

    def act(self, obs):
        obs = obs.to(self.device)
        if self.epsilon != 0 and torch.rand(1) < self.epsilon:
            t = torch.rand(self._env.out_action)
            return (t - t.min()) / (t.max() - t.min() + 1e-6)
        return self.policy(obs.unsqueeze(dim=0)).squeeze(dim=0).softmax(dim=-1)

    def trajectories(self, size: int):
        for trajectory in self.memory.trajectories(size):
            trajectory["obs"] = self._env.preprocess_fn(trajectory["obs"])
            trajectory["next_obs"] = self._env.preprocess_fn(trajectory["next_obs"])
            yield trajectory.to(self.device)

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

    def step(self):
        ...

    def learn(self) -> dict:
        ...

    def save(self):
        ...

    def load(self):
        ...

    @property
    def device(self):
        return next(self.parameters()).device

    def optimize(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, clip=None):
        optimizer.zero_grad()
        loss.backward()
        if clip is not None:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-clip, clip)
        optimizer.step()

    def train(self, mode: bool = True):
        self.epsilon.train()
        return super().train(mode)

    def eval(self, env: Environment):
        super().eval()
        self.epsilon.eval()
        self.drop(env)
