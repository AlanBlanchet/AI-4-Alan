from functools import cached_property
from typing import Generator

import numpy as np
import torch
from pydantic import BaseModel, PrivateAttr

from ...configs.base import BaseMP
from ...configs.log import Color
from ...utils.func import TensorInfo, parse_tensor
from .globals import DEFAULT_ENV_KEYS, DEFAULT_KEYS_TYPE


class ReplayStrategy(BaseModel):
    batch_size: int = 32


class TreeStrategy(ReplayStrategy): ...


class StateDict(dict[DEFAULT_KEYS_TYPE, torch.Tensor | list]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(
        self, __key: str | list | torch.Tensor
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(__key, list):
            return (super().__getitem__(k) for k in __key)
        return super().__getitem__(__key)

    def squeeze(self, dim: int):
        return StateDict(**{k: v.squeeze(dim) for k, v in self.items()})

    def format_out(self):
        """Format the observations to the correct dtype"""
        d = {}
        for k, v in self.items():
            if v.dtype == torch.uint8:
                d[k] = v.to(dtype=torch.float32) / 255.0
            else:
                d[k] = v.to(dtype=torch.float32)
        return StateDict(**d)


class DelayedInfo(BaseModel):
    map: str
    shape: list[tuple[int, ...]] = []
    info: TensorInfo


class DelayedInfos(dict[str, DelayedInfo]):
    def __init__(self, **data: dict[str, DelayedInfo]):
        super().__init__(**data)

    @cached_property
    def extras(self):
        """Key with only the extra delayed keys"""
        return DelayedInfos(**{k: v for k, v in list(self.items())[1:]})


class ReplayBuffer(BaseMP):
    """This class is a smart replay buffer that can handle delayed keys
    and retrieving special parts of the buffer. It can also act as a priority buffer by adding a priority key.

    It inherits from dict.
    You can index with multiple data types:
    - int: Get the transition at the index
    - slice: Get the transitions in the slice
    - list, np.ndarray, torch.Tensor: Get the transitions at the indexes
    - str: Get the full transition for a key, or delayed transitions
    """

    log_name = "replay buffer"
    color = Color.black

    capacity: int
    obs_info: TensorInfo = TensorInfo(shape=())
    action_info: TensorInfo = TensorInfo(shape=())
    strategy: ReplayStrategy = None

    _delayed_info: DelayedInfos = PrivateAttr(DelayedInfos())

    _states: StateDict = PrivateAttr(StateDict())

    _ptr: int = PrivateAttr(0)
    _step: int = PrivateAttr(0)

    _episode_steps: list[int] = PrivateAttr([])
    _episode_last_elements: dict[int, dict[str, torch.Tensor]] = PrivateAttr({})

    @cached_property
    def idx_dtype(self):
        if self.capacity < 2**31:
            dtype = torch.int32
        else:
            dtype = torch.int64
        self.log_debug(f"Using f{dtype=} for indexes")
        return dtype

    def model_post_init(self, __context):
        super().model_post_init(__context)

        self.reset()
        self._setup()
        self.log_info(f"Setup buffer of size {self.capacity}")

    def reset(self):
        obs = self.obs_info
        action = self.action_info
        self._states = StateDict(
            obs=torch.zeros(self.capacity, *obs.shape, dtype=obs.dtype),
            action=torch.zeros(self.capacity, *action.shape, dtype=action.dtype),
            # No next_obs, next_obs = obs.roll(-1) with next_obs[-1] = last_obs
            # And memorizing each last_obs of each episode
            **{f"{k}": torch.zeros(self.capacity) for k in ["reward", "done"]},
            idx=torch.zeros(self.capacity, dtype=self.idx_dtype),
        )

    def _setup(self):
        """Setup the buffer"""
        self._ptr = 0
        self._step = 0

        # Keys that we should roll by 1
        self._delayed_info = DelayedInfos(
            next_obs=DelayedInfo(map="obs", info=self.obs_info)
        )

        # Start an episode at _ptr
        self._episode_steps = [self._ptr]
        # Store last observation of each episode
        self._episode_last_elements: dict[int, dict[str, float]] = {}

        self._setup_last_delayed()

    def _setup_last_delayed(self):
        """Setup the last delayed elements

        For example, next_obs is delayed and samples are already in the buffer. But the last next_obs seen isn't.
        We need to store it somewhere
        """
        elems = {}
        if hasattr(self, "_next_elements"):
            elems = self._last_delayed
        self._last_delayed = {k: elems.get(k, None) for k in self._delayed_info}

    def setup_delayed(self, delayed_info: dict[str, DelayedInfo]):
        # Some keys are delayed to store the next values
        self.log_info("Setting up delayed keys", delayed_info)
        self._delayed_info.update(delayed_info)
        self._setup_last_delayed()
        for next, v in delayed_info.items():
            self._ensure_key(v.map, v.shape)
            self._last_delayed[next] = torch.zeros(*v.shape, dtype=torch.float32)

    def clear(self):
        """Clear the buffer"""
        for k in self._states.keys():
            self._states[k].zero_()
        self._setup()

    def _ensure_key(
        self, key: str, shape: tuple[int, ...] = (), dtype: torch.dtype = torch.float32
    ):
        if key not in self:
            self._states.update({key: torch.zeros(self.capacity, *shape, dtype=dtype)})

    def store(self, data: tuple):
        """Save a transition"""

        assert len(data) in [5, 6], "Data should be a tuple of 5 or 6 elements"

        if len(data) == 5:
            data += ({},)

        # Set the actual data in the buffer
        for k, v in zip(DEFAULT_ENV_KEYS, data):
            if k in self._delayed_info:  # Optimized
                continue
            self._states[k][self._ptr] = parse_tensor(v)

        # Set actual data for delayed keys in the buffer
        for k, (current, _) in zip(list(self._delayed_info.extras.values()), data[5]):
            self._states[k][self._ptr] = parse_tensor(current)

        self._states["idx"][self._ptr] = self._ptr

        if self._ptr in self._episode_steps:
            # We are erasing part of an episode. The episode should no be considered anymore
            self._episode_steps.remove(self._ptr)
        elif self._ptr in self._episode_last_elements.keys():
            # We no longer need the next_obs from that episode
            self._episode_last_elements.pop(self._ptr)

        # Watch each episode ending
        if data[4]:
            self._episode_steps.append(self._ptr)
            elems = {}
            elems.update(
                {
                    k: v
                    for k, (_, v) in zip(
                        list(self._delayed_info.extras.keys()), data[5]
                    )
                }
            )
            # Create the dict where all rolled elements will be stored
            self._episode_last_elements[self._ptr] = {"next_obs": data[3], **elems}

        self._last_delayed["next_obs"] = data[3]

        # Also do it for manually set delayed keys
        for k, (_, next) in zip(list(self._delayed_info.extras.keys()), data[5]):
            self._last_delayed[k] = next

        # Handle pointer
        if self._ptr + 1 == self.capacity:
            self._ptr = 0
        else:
            self._ptr += 1

        self._episode_steps

        # Global state
        self._step += 1

    def trajectories(
        self, size: int, history: int = 0
    ) -> Generator[StateDict, None, None]:
        """Sample an entire trajectory"""
        assert len(self._episode_steps) > 1, "No episode to sample from"
        idxs = np.random.choice(range(len(self._episode_steps[1:])), size, replace=True)

        for idx in idxs:
            sample = self._episode_steps[idx + 1]
            prev_sample = self._episode_steps[idx]

            if prev_sample > sample:
                yield self._wrap_trajectory_history(
                    self._resolve_overlap(slice(prev_sample, sample)), history
                )
            else:
                yield self._wrap_trajectory_history(self[prev_sample:sample], history)

    def _wrap_trajectory_history(
        self,
        s: StateDict,
        history: int,
        idx: slice | np.ndarray | torch.Tensor | list = None,
        cross: bool = False,
    ):
        """Wrap a trajectory with history.
        The goal is to make sure that the history is correctly handled by masking the data that should not be included.

        For instance if the history overlaps with a previously done experience, we should not include it. (set to 0)

        Args:
            s (StateDict): The state dictionary
            history (int): The amount of history to keep
            idx (slice | np.ndarray | torch.Tensor | list, optional): The indexes to keep. Defaults to None.
            cross (bool, optional): If we should cross the history. Defaults to False.
        """
        if history != 0 and not cross:
            # Create mask
            done = s["done"]
            if isinstance(idx, (np.ndarray, torch.Tensor, list)):
                idx = parse_tensor(idx)
                mask = torch.ones_like(done, dtype=torch.int32)
                dones = done.clone()
                # If we overlap with ptr
                dones[idx == self._ptr - 1] = 1
                # Chosen idx to 0 since we want it to be included
                dones[..., -1] = 0
                # Mask before done
                done_mask = dones.flip(-1).cumsum(-1).flip(-1)
                mask[done_mask > 0] = 0
                # If start
                if self._step < self.capacity:
                    mask[idx < 0] = 0
                mask = mask.bool()

            shape = mask.shape
            ndim = mask.ndim
            mask = mask.reshape(*shape, *[1 for _ in s["obs"].shape[ndim:]])
            mask = mask.expand(*shape, *s["obs"].shape[ndim:])
            # TODO remove when safe
            s["obs"][~mask] = 0
            s["next_obs"][~mask] = 0
        return s

    def last(self, history: int = 1, cross=False) -> StateDict:
        """Get the last transition with history"""
        if history == 0:
            history = 1
        stop = self._ptr
        prev = stop - history
        idx = torch.arange(prev, stop)
        return self._wrap_trajectory_history(self[idx], history, idx, cross=cross)

    def sample(self, size: int = 1, history=1) -> StateDict:
        # Get indexes
        idx = np.random.choice(range(len(self)), size, replace=True)

        if history != 0:
            idx = np.stack([idx - h for h in range(history)], axis=-1)[..., ::-1].copy()

        return self._wrap_trajectory_history(self[idx], history, idx)

    # def batched_sample(self, history: int = 0) -> Generator[StateDict, None, None]:
    #     """Sample the buffer in batches

    #     Here we can apply the segment tree to sample the buffer in a prioritized way

    #     Args:
    #         history (int, optional): The history to keep. Defaults to 0.

    #     Yields:
    #         Generator[StateDict, None, None]: The sampled batch
    #     """
    #     segment = self.tree.total / self.batch_size
    #     for i in range(self.batch_size):
    #         a, b = segment * i, segment * (i + 1)
    #         cumsum = random.uniform(a, b)

    #         tree_idx, priority, sample_idx = self.tree.get(cumsum)

    #         yield self._wrap_trajectory_history(self[idx], history, idx)

    def memorize(self, memo: dict):
        assert "idx" in memo, "idx is required to memorize a transition"
        idx = memo.pop("idx")
        for k, v in memo.items():
            if k in self._delayed_info:
                self.log_warn_once(f"Ignoring key {k} since it is delayed (not stored)")
                continue

            t = parse_tensor(v)

            if k not in self._states.keys():
                self._ensure_key(k, t.shape[1:], t.dtype)

            self._states[k][idx] = t

    def __getitem__(self, __key: DEFAULT_KEYS_TYPE | slice | list[int]):
        if isinstance(__key, (slice, int)):
            d = StateDict()
            # Default stored keys
            for k, v in self._states.items():
                d.update({k: v[__key]})
            # Add delayed unstored keys
            for k in self._delayed_info:
                d.update({k: self._get_delayed_elements(k, __key)})
            return d

        if isinstance(__key, (list, np.ndarray, torch.Tensor)):
            __key = parse_tensor(__key, dtype=torch.int32)
            d = StateDict()
            for k, v in self._states.items():
                v = v[__key.flatten()].view(*__key.shape, *v.shape[1:])
                d.update({k: v})
            for k in self._delayed_info:
                d.update({k: self._get_delayed_elements(k, __key)})
            return d

        if self._step < self.capacity:
            return self._get_delayed_elements(__key, slice(self._ptr))
        return self._get_delayed_elements(__key).roll(-self._ptr, 0).flip(0)

    def _get_delayed_elements(self, __key: str, s: slice | torch.Tensor | int = None):
        """Get the delayed elements real values
        This method uses the stored next values with the next key and create the tensor for the current delayed key.

        For example in a default environment, the obs key is delayed and the next_obs is stored in the next_obs key.
        Thus when we try to get the obs key, we will get the next_obs key rolled by 1.

        Args:
            __key (str): The key to get the delayed elements
            s (slice | torch.Tensor | int, optional): The slice or index to get. Defaults to None.
        """
        if __key in self._delayed_info:
            info = self._delayed_info[__key]
            original = self._states[info.map]
            if s is None:
                next = original.roll(-1, 0)
                list(self._delayed_info).index(__key)
                next[-1] = parse_tensor(self._last_delayed[__key])
                if len(self._episode_last_elements) > 0:
                    # Override final obs with last obs of each episode
                    next[list(self._episode_last_elements.keys())] = parse_tensor(
                        [v[__key] for v in self._episode_last_elements.values()],
                        dtype=info.info.dtype,
                    )
                return next
            else:
                if isinstance(s, slice):
                    next = original[s.start : s.stop].roll(-1, 0)
                    replace = (
                        parse_tensor(self._last_delayed[__key])
                        if s.stop - 1 == self._ptr
                        else original[self._idx_parse(s.stop)]
                    )
                    next[-1] = replace
                    idx = torch.arange(s.start, s.stop)
                    available = {
                        k - s.start: v[__key]
                        for k, v in self._episode_last_elements.items()
                        if k in idx
                    }
                    next[list(available.keys())] = parse_tensor(
                        list(available.values()), dtype=info.info.dtype
                    )
                    return next
                elif isinstance(s, (int, torch.Tensor)):
                    next_s = self._idx_parse(s + 1)
                    if isinstance(next_s, int):
                        next = original[next_s]
                        if s in self._episode_last_elements:
                            next[s] = parse_tensor(
                                self._episode_last_elements[s][__key],
                                dtype=info.info.dtype,
                            )
                    else:
                        next = original[next_s.flatten()]
                        for k, v in self._episode_last_elements.items():
                            if k in s:
                                idx = (s.flatten() == k).argwhere().flatten()
                                next[idx] = parse_tensor(
                                    v[__key], dtype=info.info.dtype
                                )
                        next = next.view(*next_s.shape, *original.shape[1:])
                    next[next_s == self._ptr] = parse_tensor(
                        self._last_delayed[__key], dtype=info.info.dtype
                    )
                    return next

        v = self._states[__key]
        return v if s is None else v[s.start : s.stop]

    def _resolve_overlap(self, s: slice, key: str = None):
        """"""
        keys = []
        if key is None:
            keys = DEFAULT_ENV_KEYS
        else:
            keys = [key]

        d = StateDict()
        for k in keys:
            if s.start < s.stop:
                d.update({k: self._get_delayed_elements(k, s)})
            else:
                slide = self.capacity - s.start - 1
                t = self._get_delayed_elements(k).roll(slide, 0)[: s.stop + slide + 1]
                d.update({k: t})

        if key is not None:
            return d[key]
        return d

    def _idx_parse(self, idx: int | torch.Tensor):
        if isinstance(idx, int):
            if idx < 0:
                idx += self.capacity
            elif idx >= self.capacity:
                idx -= self.capacity
        elif isinstance(idx, torch.Tensor):
            idx[idx < 0] += self.capacity
            idx[idx >= self.capacity] -= self.capacity
        return idx

    def __len__(self):
        return min(self._step, self.capacity)
