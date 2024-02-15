import random
from abc import ABC
from collections import deque
from itertools import islice
from typing import Generator, Iterator

import numpy as np
import torch
import torch.multiprocessing as mp
from easydict import EasyDict as edict
from torch.distributions import Categorical

from ..utils.func import parse_tensor
from .keys import DEFAULT_ENV_KEYS, DEFAULT_KEYS_TYPE
from .state import StateDict


class SmartReplayBuffer(dict[DEFAULT_KEYS_TYPE | str, torch.Tensor]):
    def __init__(self, capacity: int, obs_shape: tuple[int], out_action: int) -> None:
        super().__init__(self.default_dict(capacity, obs_shape, out_action))

        self._capacity = capacity
        self._pointer = 0

        self.step_ = 0
        # Start an episode at _pointer
        self.episode_steps_ = [self._pointer]
        self.next_obs = None

    def _next_obs_map(self, obs: torch.Tensor, end: int):
        obs = obs.roll(-1, 0)
        if end == self._capacity - 1:
            # TODO
            ...
        else:
            obs[-1] = self.__getitem__("obs")[end - 1]

    @classmethod
    def default_dict(
        cls, capacity: int, obs_shape: tuple[int], out_action: int
    ) -> edict[DEFAULT_KEYS_TYPE | str, torch.Tensor]:
        return edict(
            obs=torch.zeros(capacity, *obs_shape, dtype=torch.float32),
            action=torch.zeros(capacity, out_action, dtype=torch.float32),
            # No next_obs, next_obs = obs.roll(-1) with with next_obs[-1] = last_obs
            **{f"{k}": torch.zeros(capacity) for k in ["reward", "done"]},
        )

    def store(self, data: tuple):
        """Save a transition"""
        # Set the actual data in the buffer
        for k, v in zip(DEFAULT_ENV_KEYS, data):
            if k == "next_obs":  # Optimized
                continue
            super().__getitem__(k)[self._pointer] = parse_tensor(v)

        # We are erasing part of an episode. The episode should no be considered anymore
        if self._pointer in self.episode_steps_:
            self.episode_steps_.remove(self._pointer)

        # Watch each episode ending
        if data[-1]:
            self.episode_steps_.append(self._pointer)

        self.next_obs = data[-2]

        # Handle pointer
        if self._pointer + 1 == self._capacity:
            self._pointer = 0
        else:
            self._pointer += 1

        self.episode_steps_
        # Global state
        self.step_ += 1

    def trajectories(self, size: int) -> Generator[StateDict, None, None]:
        idxs = np.random.choice(range(len(self.episode_steps_[1:])), size, replace=True)

        for idx in idxs:
            sample = self.episode_steps_[idx + 1]
            prev_sample = self.episode_steps_[idx]

            if prev_sample > sample:
                yield self._resolve_overlap(slice(prev_sample, sample))
            else:
                yield self[prev_sample:sample]

    def last(self) -> StateDict:
        return StateDict({k: v[self._pointer - 1] for k, v in super().items()})

    def __getitem__(self, __key: DEFAULT_KEYS_TYPE | slice):
        if isinstance(__key, slice):
            d = StateDict()
            for k, v in super().items():
                d.update({k: v[__key]})
            d.update({"next_obs": self._get_item_slice("next_obs", __key)})
            return d

        if self.step_ < self._capacity:
            return self._get_item_slice(__key, slice(self._pointer))
        return self._get_item_slice(__key).roll(-self._pointer, 0).flip(0)

    def _get_item_slice(self, __key: str, s: slice = None):
        if __key == "next_obs":
            obs = super().__getitem__("obs")
            if s is None:
                next_obs = obs.roll(-1, 0)
                next_obs[-1] = parse_tensor(self.next_obs)
                return next_obs
            else:
                next_obs = obs[s.start : s.stop].roll(-1, 0)
                replace = (
                    parse_tensor(self.next_obs)
                    if s.stop == self._pointer
                    else obs[s.stop]
                )
                next_obs[-1] = replace
                return next_obs

        v = super().__getitem__(__key)
        return v if s is None else v[s.start : s.stop]

    def _resolve_overlap(self, s: slice, key: str = None):
        keys = []
        if key is None:
            keys = DEFAULT_ENV_KEYS
        else:
            keys = [key]

        d = StateDict()
        for k in keys:
            if s.start < s.stop:
                d.update({k: self._get_item_slice(k, s)})
            else:
                slide = self._capacity - s.start - 1
                t = self._get_item_slice(k).roll(slide, 0)[: s.stop + slide + 1]
                d.update({k: t})

        if key is not None:
            return d[key]
        return d

    def __len__(self):
        return min(self.step_, self._capacity)


class BaseBuffer(ABC):
    @property
    def buffer_count(self):
        return len(self.buffer)

    def __init__(self) -> None:
        self.buffer_sample_count = 0

    def push(self, *args):
        """Save a transition"""
        self.buffer.append(args)

    def reset(self):
        self.buffer.clear()
        self.buffer_sample_count = 0

    def sample(self, batch_size: int, stacks=1):
        idx = random.sample(range(len(self) - stacks + 1), batch_size)

        # [zip[s, a, r, s', d], ...]
        item_stacks = [zip(*self[i : i + stacks]) for i in idx]

        self.buffer_sample_count += 1
        return item_stacks

    def extract(self, num: int = None):
        if num is None:
            return zip(*self.buffer)
        return zip(*self[:num])

    def slice(self, start: int, end: int) -> Iterator[torch.Tensor]:
        return zip(*self[start:end])

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            key = list(islice(list(range(len(self))), *key.indices(len(self))))
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"The index {key} is out of range.")
            return self.buffer[key]

        if isinstance(key, list):
            return [self.buffer[i] for i in key]

        raise TypeError("Invalid argument type.")

    def __lt__(self, other):
        return self.buffer_count < other

    def __rlt__(self, other):
        return other < self.buffer_count

    def __le__(self, other):
        return self.buffer_count <= other

    def __rle__(self, other):
        return other <= self.buffer_count

    def __ge__(self, other):
        return self.buffer_count >= other

    def __rge__(self, other):
        return other >= self.buffer_count

    def __gt__(self, other):
        return self.buffer_count > other

    def __rgt__(self, other):
        return other > self.buffer_count

    def __eq__(self, other):
        return self.buffer_count == other

    def __req__(self, other):
        return other == self.buffer_count

    def __len__(self):
        return len(self.buffer)


class Buffer(BaseBuffer):
    def __init__(self):
        super().__init__()
        self.buffer = []


class DequeueBuffer(BaseBuffer):
    def __init__(self, size: int = 10000):
        super().__init__()
        self.size = size
        self.buffer: deque[tuple[torch.Tensor]] = self._create_deque()

    def _create_deque(self):
        return deque([], maxlen=self.size)


class SharedStateBuffer:
    def __init__(self, trajectory_size: int):
        self.traj_size = trajectory_size
        # Store trajectories and priorities
        self.buffer = self._create_deque(trajectory_size)

    def _create_deque(self, size: int):
        return mp.Queue()

    def push(self, trajectory):
        """Save a trajectory"""
        self.buffer.put(trajectory)


class StateBuffer:
    def __init__(self, trajectory_size: int):
        self.traj_size = trajectory_size
        # Store trajectories and priorities
        self.buffer = self._create_deque(trajectory_size)
        self.priorities = self._create_deque(trajectory_size)

    def reset(self):
        # To match api code is really bad :(
        ...

    def _create_deque(self, size: int):
        return deque([], maxlen=size)

    def push(self, trajectory):
        """Save a trajectory"""
        # Set high priority for new trajectories
        self.priorities.append(50)
        self.buffer.append(trajectory)

    def sample(self, states: int) -> tuple[torch.Tensor, list[StateDict]]:
        """Sample a batch of trajectories"""
        distrib = Categorical(torch.tensor(self.priorities))
        idx = distrib.sample((states,))
        return idx, [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)
