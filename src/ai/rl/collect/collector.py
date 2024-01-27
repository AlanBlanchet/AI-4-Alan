from __future__ import annotations

import random
from typing import TYPE_CHECKING, Literal

import torch
from gymnasium.spaces import Discrete
from tensorboardX import SummaryWriter

from ..utils.func import parse_tensor
from ..utils.hyperparam import Hyperparam, to_hyperparam
from .buffer import DequeueBuffer
from .environment import Environment

if TYPE_CHECKING:
    from ..policy import Policy

TRANSITIONS = tuple[torch.Tensor, ...]
COLLECTOR_MODES = Literal["reset", "keep"]
COLLECTION_OF = Literal["steps", "episodes"]
COLLECTOR_RANDOM_TYPE = Literal["step", "episode"]


class Collector(Environment):
    def __init__(
        self,
        env_id: str,
        collect=4,
        of: COLLECTION_OF = "episodes",
        epsilon: int | Hyperparam = 0.9,
        buffer_size: int = 10000,
        video_freq=500,
    ):
        self.buffer_size = buffer_size
        super().__init__(env_id)
        self.default_collect = collect
        self.default_of = of
        self.current_state = None
        self._n_episodes = 0
        self._episode_start = 0
        self._n_steps = 0
        self.logger: SummaryWriter = None
        self.mode = "reset"
        self.epsilon = to_hyperparam(epsilon)
        self._train = True
        self.buffer = DequeueBuffer(self.buffer_size)
        self.episode_buffer = []
        self.video_freq = video_freq

        assert of in ["steps", "episodes"], f"Invalid of value: {of}"

    def set_mode(self, mode: COLLECTOR_MODES):
        self.mode = mode

    def set_policy(self, policy: Policy):
        self._policy = policy

    def fill(self, n: int):
        """
        Fills the buffer with n steps.
        """
        return self.collect_steps(n)

    def sample(
        self,
        n: int,
        device: torch.device,
        stacks=1,
        rand_type: COLLECTOR_RANDOM_TYPE = "step",
    ) -> TRANSITIONS:
        """
        Samples n steps from the buffer.
        """
        data = None
        if rand_type == "step":
            # TODO make sure we don't sample from 2 diffent episodes when stacking
            stack_data = self.buffer.sample(n, stacks=stacks)
            # stack = [(s,s,s), (s,s,s), ...]
            data = [list(sum(stack, ())) for stack in zip(*stack_data)]  # Flatten stack
        elif rand_type == "episode":
            trajectories_data = [self._sample_episode_steps(n) for _ in range(stacks)]
            data = [torch.cat(traj) for traj in zip(*trajectories_data)]
        else:
            raise ValueError(f"Invalid random type: {rand_type}")

        tensor_data = [parse_tensor(x, device=device) for x in data]
        return tensor_data

    def _sample_episode_steps(self, n: int):
        # Use episode buffer to get max(n_steps) from an episode
        # We allow getting less than n_steps if the episode is less than n_steps.
        # Otherwise we chose a position in the episode to start from.
        episode_buffer = torch.tensor(self.episode_buffer)
        # Get indexes of episodes from data buffer
        buffer_idx = episode_buffer.flip(0).cumsum(0)
        # step_diff = self._n_steps - buffer_idx[-1]
        if self._n_steps < self.buffer_size:
            buffer_idx = torch.cat([buffer_idx, torch.tensor([self._n_steps])])
        # Select 2 neighbor points from buffer_idx
        r = random.randint(0, len(buffer_idx) - 2)
        start, end = buffer_idx[r], buffer_idx[r + 1]
        # Select random starting point in episode
        r = random.randint(0, max(0, end - start - n))
        start += r
        data = self.buffer.slice(start, min(start + n, end))
        data = [parse_tensor(x, "cuda") for x in data]
        _data: list[torch.Tensor] = []
        # Make sure it is of batch_size n
        for d in data:
            target = torch.zeros(n, *d.shape[1:])
            target[n - d.shape[0] :] = d
            _data.append(target)
        return _data

    def collect(self, device: torch.device):
        """
        Collects defined amount of steps/episodes from the environment using the given policy.
        """
        if self.default_of == "steps":
            raise NotImplementedError()
        elif self.default_of == "episodes":
            return self.collect_episodes(self.default_collect, device)

    def collect_episodes(self, n_episodes: int, device: torch.device):
        """
        Collects n episodes from the environment using the given policy.
        """
        if self.mode == "reset":
            self.buffer.reset()

        stored_episodes: list[TRANSITIONS] = []
        for _ in range(n_episodes):
            done = False
            steps = 0
            while done is False:
                *_, done = self.collect_step()
                steps += 1

            # Optimization
            steps = steps if self.mode == "keep" else None

            data = self._extract(device, steps)

            stored_episodes.append(data)
            self.logger.add_scalar(
                "collector/episodes", self._n_episodes, self._n_steps
            )

        return stored_episodes

    def _extract(self, device: torch.device = None, num: int = None):
        device = device or torch.device("cpu")
        data = self.buffer.extract(num)
        tensor_data = [parse_tensor(x, device=device) for x in data]
        if self.mode == "reset":
            self.buffer.reset()
        return tensor_data

    def collect_steps(self, n_steps: int, device: torch.device = None):
        """
        Collects n steps from the environment using the given policy.
        """
        if self.mode == "reset":
            self.buffer.reset()

        for _ in range(n_steps):
            self.collect_step()

        return self._extract(device, n_steps)

    def collect_step(self, store=True):
        if self.current_state is None:
            self.current_state = torch.from_numpy(self._env.reset()[0]).float()

        state = self.current_state

        # We sometimes need to prefill the buffer at random (ex: stacks)
        requires_gready = self._policy.stacked and self._n_steps <= self._policy.stacks

        action = None
        epsilon = self.epsilon.step(self._n_steps)
        self.logger.add_scalar("collector/epsilon", epsilon, self._n_steps)

        # Collect epsilon greedy action
        if (self._train and torch.rand(1) < epsilon) or requires_gready:
            if isinstance(self._env.action_space, Discrete):
                action = torch.randn(self._env.action_space.n)
            else:
                action = torch.randint(self._env.action_space.n)

        if action is None:
            # Sample from policy
            if self._policy.stacked:
                # Extract num previous states
                s, *_ = self._extract(num=self._policy.stacks)
                action = self._policy.batch_act(s).squeeze(dim=0).detach()
            else:
                action = self._policy.act(state)

        assert action is not None, "Action cannot be None"

        # Parse action to be in the format of the env
        action = action.cpu()
        parsed_action = None
        if isinstance(self._env.action_space, Discrete):
            parsed_action = action.argmax().numpy()
        else:
            parsed_action = action.numpy()

        assert (
            state.device == action.device
        ), f"State ({state.device}) and action ({action.device}) must be on the same device"

        next_state, reward, terminated, truncated, _ = self._env.step(parsed_action)

        done = terminated or truncated

        self._n_steps += 1
        self._episode_start += 1

        if done:
            self._n_episodes += 1
            self.episode_buffer.append(self._episode_start)
            self._episode_start = 0
            self.current_state = None
            self.logger.add_scalar(
                "collector/episodes", self._n_episodes, self._n_steps
            )
            self.logger.add_scalar(
                "collector/buffer", len(self.buffer) + 1, self._n_steps
            )
        else:
            self.current_state = torch.from_numpy(next_state).float()

        data = (state, action, reward, next_state, done)
        if store:
            self.buffer.push(*data)

        if self._n_steps % self.video_freq == 0:
            self._log_video()

        return data

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def set_render(self, render_mode: str):
        super().set_render(render_mode)
        self.current_state = None
        return self

    def set_logger(self, logger: SummaryWriter):
        self.logger = logger

    def __getstate__(self) -> object:
        state = super().__getstate__()
        del state["logger"]
        del state["_policy"]
        return state

    def _log_video(self):
        s, *_ = self._extract(num=64)

        s = s.permute(0, -1, -3, -2).unsqueeze(dim=0)

        self.logger.add_video(
            "collector/video",
            s,
            self._n_steps,
            fps=60,
        )
