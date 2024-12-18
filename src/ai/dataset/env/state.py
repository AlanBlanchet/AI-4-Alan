from __future__ import annotations

from typing import Literal

import torch

from .globals import DEFAULT_ENV_KEYS, DEFAULT_KEYS_TYPE


class StateDict(dict[DEFAULT_KEYS_TYPE, torch.Tensor | list]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_defaults(self, *extra_keys: list[str]):
        return [self.__getitem__(key) for key in [*DEFAULT_ENV_KEYS, *extra_keys]]

    def __getitem__(
        self,
        __key: Literal["obs", "action", "reward", "next_obs", "done", "idx"] | list,
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(__key, list):
            return (super(StateDict, self).__getitem__(k) for k in __key)
        return super().__getitem__(__key)

    def to(self, device: str):
        for k in self.keys():
            self[k] = self[k].to(device)
        return self

    def squeeze(self, dim: int):
        return StateDict(**{k: v.squeeze(dim) for k, v in self.items()})

    # def episode_split(self):
    #     """
    #     Splits the current state into a list of states each containing terminated episodes and padding
    #     Creates a mask to be used for the loss function
    #     """
    #     states: list[StateDict] = []
    #     # Check if mask is already present. In that case the split has already been done
    #     if "mask" in self.keys():
    #         states.append(self)
    #         return states

    #     dones = self["done"]
    #     n = len(dones)
    #     dones_idx = (dones == 1).nonzero().flatten()

    #     if len(dones_idx) == 0:
    #         mask = torch.ones(n)
    #         self["mask"] = mask
    #         states.append(self)
    #         return states

    #     # Split each dones
    #     dones_idx += 1
    #     if dones_idx[-1] != n:
    #         dones_idx = torch.cat([dones_idx, torch.tensor([n])])
    #     dones_idx = torch.cat([torch.tensor([0]), dones_idx])
    #     dones_sizes = [
    #         len(dones[dones_idx[i] : dones_idx[i + 1]])
    #         for i in range(len(dones_idx) - 1)
    #     ]

    #     if len(dones_sizes) == 1:
    #         mask = torch.ones(n)
    #         self["mask"] = mask
    #         states.append(self)
    #         return states

    #     splitted_states = [torch.split(v, dones_sizes) for v in self.values()]

    #     for item_states_tuple in zip(*splitted_states):
    #         # One mask for every splits
    #         mask = torch.zeros(n)
    #         split_len = len(item_states_tuple[0])
    #         new_state_list = []
    #         for item in item_states_tuple:
    #             padded = torch.zeros(n, *item.shape[1:])
    #             padded[:split_len] = item
    #             new_state_list.append(padded)
    #         mask[:split_len] = 1
    #         new_state = StateDict()
    #         new_state.add_defaults(new_state_list)
    #         new_state["mask"] = mask
    #         new_state._to_tensors()
    #         states.append(new_state)

    #     return states

    # @staticmethod
    # def batch(states: list[StateDict]):
    #     """
    #     Batch a list of states into a single state
    #     """
    #     batched_state = StateDict()

    #     for key in states[0].keys():
    #         # batched_state[key] = torch.stack([state[key] for state in states])
    #         batched_state[key] = torch.cat([state[key] for state in states])
    #     return batched_state

    def __repr__(self) -> str:
        max_str_len = max([len(k) for k in self.keys()]) + 1
        vals = ",\n  ".join(
            [f"{k.ljust(max_str_len, ' ')} = {self[k].shape}" for k in self.keys()]
        )
        return f"StateDict(\n  {vals}\n)"
