from typing import Generator

import numpy as np
import torch
from torch.distributions import Categorical

from ..utils.func import parse_tensor
from .keys import DEFAULT_ENV_KEYS, DEFAULT_KEYS_TYPE
from .state import StateDict


class SmartReplayBuffer(dict[DEFAULT_KEYS_TYPE | str, torch.Tensor]):
    def __init__(self, capacity: int, obs_shape: tuple[int], out_action: int) -> None:
        super().__init__(
            self.default_dict(
                capacity, obs_shape, (out_action,) if out_action > 0 else ()
            )
        )

        self._setup(capacity)

    def _setup(self, capacity: int = None):
        if capacity is not None:
            self._capacity = capacity
        self._ptr = 0

        self.step_ = 0
        delayed_key_maps = {"next_obs": "obs"}
        if hasattr(self, "delayed_key_map"):
            delayed_key_maps.update(self.delayed_key_map)
        delayed_shapes = []
        if hasattr(self, "delayed_shapes"):
            delayed_shapes = self.delayed_shapes
        # Keys that we should roll by 1
        self.delayed_key_map = delayed_key_maps
        self.delayed_shapes = delayed_shapes
        # Start an episode at _pointer
        self.episode_steps_ = [self._ptr]
        # Store last observation of each episode or other
        self.episode_last_elements: dict[int, dict[str, float]] = {}
        self._setup_elements()

    def _setup_elements(self):
        elems = {}
        if hasattr(self, "next_elements"):
            elems = self.next_elements
        self.next_elements = {
            k: elems.get(k, None) for k in self.delayed_key_map.keys()
        }

    def setup_delayed(
        self, delayed_key_map: dict[str, str], shapes: list[tuple[int, ...]]
    ):
        # Some keys are delayed to store the next values
        self.delayed_key_map = {**self.delayed_key_map, **delayed_key_map}
        self.delayed_shapes = shapes
        self._setup_elements()
        for (next, current), shape in zip(delayed_key_map.items(), shapes):
            self._ensure_key(current, shape)
            self.next_elements[next] = torch.zeros(*shape, dtype=torch.float32)

    @classmethod
    def default_dict(
        cls, capacity: int, obs_shape: tuple[int], out_action: int
    ) -> dict[DEFAULT_KEYS_TYPE | str, torch.Tensor]:
        return dict(
            obs=torch.zeros(capacity, *obs_shape, dtype=torch.float32),
            action=torch.zeros(capacity, *out_action, dtype=torch.float32),
            # No next_obs, next_obs = obs.roll(-1) with next_obs[-1] = last_obs
            # And memorizing each last_obs of each episode
            **{f"{k}": torch.zeros(capacity) for k in ["reward", "done"]},
            idx=torch.zeros(capacity, dtype=torch.int32),
        )

    def clear(self):
        for k in super().keys():
            if k == "p":
                super().__getitem__(k).fill_(1)
            else:
                super().__getitem__(k).zero_()
        self._setup()

    def _ensure_key(
        self, key: str, shape: tuple[int, ...] = (), dtype: torch.dtype = torch.float32
    ):
        if key not in self:
            tfunc = torch.ones if key == "p" else torch.zeros
            super().update({key: tfunc(self._capacity, *shape, dtype=dtype)})

    def store(self, data: tuple):
        """Save a transition"""
        extra = len(data) > 5

        # Set the actual data in the buffer
        for k, v in zip(DEFAULT_ENV_KEYS, data):
            if k in self.delayed_key_map:  # Optimized
                continue
            super().__getitem__(k)[self._ptr] = parse_tensor(v)

        if extra:
            # Set actual data for delayed keys in the buffer
            for k, (current, _) in zip(
                list(self.delayed_key_map.values())[1:], data[5]
            ):
                super().__getitem__(k)[self._ptr] = parse_tensor(current)

        if "p" in super().keys():
            super().__getitem__("p")[self._ptr] = 1

        super().__getitem__("idx")[self._ptr] = self._ptr

        if self._ptr in self.episode_steps_:
            # We are erasing part of an episode. The episode should no be considered anymore
            self.episode_steps_.remove(self._ptr)
        elif self._ptr in self.episode_last_elements.keys():
            # We no longer need the next_obs from that episode
            self.episode_last_elements.pop(self._ptr)

        # Watch each episode ending
        if data[4]:
            self.episode_steps_.append(self._ptr)
            elems = {}
            if extra:
                elems.update(
                    {
                        k: v
                        for k, (_, v) in zip(
                            list(self.delayed_key_map.keys())[1:], data[5]
                        )
                    }
                )
            # Create the dict where all rolled elements will be stored
            self.episode_last_elements[self._ptr] = {"next_obs": data[3], **elems}

        self.next_elements["next_obs"] = data[3]

        if extra:
            # Also do it for manually set delayed keys
            for k, (_, next) in zip(list(self.delayed_key_map.keys())[1:], data[5]):
                self.next_elements[k] = next

        # Handle pointer
        if self._ptr + 1 == self._capacity:
            self._ptr = 0
        else:
            self._ptr += 1

        self.episode_steps_
        # Global state
        self.step_ += 1

    def trajectories(
        self, size: int, history: int = 0
    ) -> Generator[StateDict, None, None]:
        assert len(self.episode_steps_) > 1, "No episode to sample from"
        idxs = np.random.choice(range(len(self.episode_steps_[1:])), size, replace=True)

        for idx in idxs:
            sample = self.episode_steps_[idx + 1]
            prev_sample = self.episode_steps_[idx]

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
    ):
        if history != 0:
            # Create mask
            done = s["done"]
            if isinstance(idx, (np.ndarray, torch.Tensor)):
                idx = parse_tensor(idx, dtype=torch.int32)
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
                if self.step_ < self._capacity:
                    mask[idx < 0] = 0
                s["mask"] = mask.bool()

            mask = s["mask"]
            shape = mask.shape
            ndim = mask.ndim
            mask = mask.reshape(*shape, *[1 for _ in s["obs"].shape[ndim:]])
            mask = mask.expand(*shape, *s["obs"].shape[ndim:])
            # TODO remove when safe
            s["obs"][~mask] = 0
            s["next_obs"][~mask] = 0

        return s

    def last(self, history: int = 0) -> StateDict:
        stop = self._ptr
        if history == 0:
            return self[stop - 1]
        prev = stop - history
        idx = torch.arange(prev, stop)
        return self._wrap_trajectory_history(self[idx], history, idx)

    def sample(self, size: int, history=0, priority: str = None) -> StateDict:
        # Get indexes
        if priority is not None:
            self._ensure_key(priority)

            samples = super().__getitem__(priority)[: len(self)]
            distrib = Categorical(samples)
            idx = distrib.sample((size,))
        else:
            idx = np.random.choice(range(len(self)), size, replace=True)

        if history != 0:
            idx = np.stack([idx - h for h in range(history)], axis=-1)[..., ::-1].copy()

        return self._wrap_trajectory_history(self[idx], history, idx)

    def memorize(self, memo: dict):
        assert "idx" in memo, "idx is required to memorize a transition"
        idx = memo.pop("idx")
        for k, v in memo.items():
            if k in self.delayed_key_map:
                print(f"Ignoring key {k} since it is delayed (not stored)")
                continue

            t = parse_tensor(v)

            if k not in super().keys():
                self._ensure_key(k, t.shape[1:], t.dtype)

            super().__getitem__(k)[idx] = t

    def __getitem__(self, __key: DEFAULT_KEYS_TYPE | slice | list[int]):
        if isinstance(__key, (slice, int)):
            d = StateDict()
            # Default stored keys
            for k, v in super().items():
                d.update({k: v[__key]})
            # Add delayed unstored keys
            for k in self.delayed_key_map.keys():
                d.update({k: self._get_rolled_elements(k, __key)})
            return d

        if isinstance(__key, (list, np.ndarray, torch.Tensor)):
            __key = parse_tensor(__key, dtype=torch.int32)
            d = StateDict()
            for k, v in super().items():
                v = v[__key.flatten()].view(*__key.shape, *v.shape[1:])
                d.update({k: v})
            for k in self.delayed_key_map.keys():
                d.update({k: self._get_rolled_elements(k, __key)})
            return d

        if self.step_ < self._capacity:
            return self._get_rolled_elements(__key, slice(self._ptr))
        return self._get_rolled_elements(__key).roll(-self._ptr, 0).flip(0)

    def _get_rolled_elements(self, __key: str, s: slice | torch.Tensor | int = None):
        if __key in self.delayed_key_map:
            mapping = self.delayed_key_map[__key]
            original = super().__getitem__(mapping)
            if s is None:
                next = original.roll(-1, 0)
                list(self.delayed_key_map.keys()).index(__key)
                next[-1] = parse_tensor(self.next_elements[__key])
                if len(self.episode_last_elements) > 0:
                    # Override final obs with last obs of each episode
                    next[list(self.episode_last_elements.keys())] = parse_tensor(
                        [v[__key] for v in self.episode_last_elements.values()]
                    )
                return next
            else:
                if isinstance(s, slice):
                    next = original[s.start : s.stop].roll(-1, 0)
                    replace = (
                        parse_tensor(self.next_elements[__key])
                        if s.stop - 1 == self._ptr
                        else original[self._idx_parse(s.stop)]
                    )
                    next[-1] = replace
                    idx = torch.arange(s.start, s.stop)
                    available = {
                        k - s.start: v[__key]
                        for k, v in self.episode_last_elements.items()
                        if k in idx
                    }
                    next[list(available.keys())] = parse_tensor(
                        list(available.values())
                    )
                    return next
                elif isinstance(s, (int, torch.Tensor)):
                    next_s = self._idx_parse(s + 1)
                    if isinstance(next_s, int):
                        next = original[next_s]
                        if s in self.episode_last_elements:
                            next[s] = parse_tensor(self.episode_last_elements[s][__key])
                    else:
                        next = original[next_s.flatten()]
                        for k, v in self.episode_last_elements.items():
                            if k in s:
                                idx = (s.flatten() == k).argwhere().flatten()
                                next[idx] = parse_tensor(v[__key])
                        next = next.view(*next_s.shape, *original.shape[1:])
                    next[next_s == self._ptr] = parse_tensor(self.next_elements[__key])
                    return next

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
                d.update({k: self._get_rolled_elements(k, s)})
            else:
                slide = self._capacity - s.start - 1
                t = self._get_rolled_elements(k).roll(slide, 0)[: s.stop + slide + 1]
                d.update({k: t})

        if key is not None:
            return d[key]
        return d

    def _idx_parse(self, idx: int | torch.Tensor):
        if isinstance(idx, int):
            if idx < 0:
                idx += self._capacity
            elif idx >= self._capacity:
                idx -= self._capacity
        elif isinstance(idx, torch.Tensor):
            idx[idx < 0] += self._capacity
            idx[idx >= self._capacity] -= self._capacity
        return idx

    def __len__(self):
        return min(self.step_, self._capacity)
