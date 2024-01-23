from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from gymnasium.spaces import Discrete
from tensorboardX import SummaryWriter

from ..utils.func import parse_tensor
from .buffer import DequeueBuffer
from .environment import Environment

if TYPE_CHECKING:
    from ..policy import Policy


class Collector(Environment):
    def __init__(
        self, env_id: str, collect=4, of: Literal["steps", "episodes"] = "episodes"
    ):
        super().__init__(env_id)
        self.buffer = DequeueBuffer()
        self.default_collect = collect
        self.default_of = of
        self.current_state = None
        self._n_episodes = 0
        self._n_steps = 0
        self.logger: SummaryWriter = None

        assert of in ["steps", "episodes"], f"Invalid of value: {of}"

    @property
    def state_shape(self):
        if isinstance(self._env.observation_space, Discrete):
            return self._env.observation_space.n
        else:
            return self._env.observation_space.shape[0]

    @property
    def action_shape(self):
        if isinstance(self._env.action_space, Discrete):
            return self._env.action_space.n
        else:
            return self._env.action_space.shape[0]

    def clean(self):
        self._env.close()

    def collect(self, policy: Policy, device: torch.device):
        """
        Collects defined amount of steps/episodes from the environment using the given policy.
        """
        if self.default_of == "steps":
            raise NotImplementedError()
        elif self.default_of == "episodes":
            return self.collect_episodes(policy, self.default_collect, device)

    def collect_episodes(self, policy: Policy, n_episodes: int, device: torch.device):
        """
        Collects n episodes from the environment using the given policy.
        """
        self.buffer.reset()
        stored_episodes: list[tuple[torch.Tensor, ...]] = []
        for _ in range(n_episodes):
            done = False
            while done is False:
                *_, done = self.collect_step(policy)

            self._n_episodes += 1
            data = self._extract(device)

            stored_episodes.append(data)
            self.logger.add_scalar(
                "collector/episodes", self._n_episodes, self._n_steps
            )

        return stored_episodes

    def _extract(self, device: torch.device):
        data = self.buffer.extract()
        tensor_data = [parse_tensor(x, device=device) for x in data]
        self.buffer.reset()
        return tensor_data

    def collect_steps(self, policy: Policy, n_steps: int):
        """
        Collects n steps from the environment using the given policy.
        """
        self.buffer.reset()
        for _ in range(n_steps):
            self.collect_step(policy)

        return self.buffer.reset()

    def collect_step(self, policy: Policy, store=True):
        if self.current_state is None:
            self.current_state = torch.from_numpy(self._env.reset()[0]).float()

        state = self.current_state
        action = policy.act(state)

        assert action is not None, "Action cannot be None"

        parsed_action = None
        if isinstance(self._env.action_space, Discrete):
            parsed_action = action.cpu().argmax().numpy()
        else:
            parsed_action = action.cpu().numpy()

        next_state, reward, terminated, truncated, _ = self._env.step(parsed_action)

        done = terminated or truncated

        if done:
            self.current_state = None
            self.logger.add_scalar(
                "collector/buffer", len(self.buffer) + 1, self._n_steps
            )
        else:
            self.current_state = torch.from_numpy(next_state).float()

        self._n_steps += 1

        if action is None:
            raise ValueError("Action cannot be None")

        data = (state, action, reward, next_state, done)
        if store:
            self.buffer.push(*data)
        return data

    def set_render(self, render_mode: str):
        super().set_render(render_mode)
        self.current_state = None
        return self

    def set_logger(self, logger: SummaryWriter):
        self.logger = logger
