# from abc import abstractmethod
# from queue import Empty
# from typing import Any, Mapping, TypeVar

# import numpy as np
# import torch
# import torch.nn as nn

# from ...dataset.env import Environment, EnvironmentDataset, StateDict
# from ...dataset.env.queues import RLQueues
# from ...utils.hyperparam import HYPERPARAM, Hyperparam
# from ...utils.pydantic_ import validator
# from .module import Module

# T = TypeVar("T")


# class Agent(Module, buildable=False):
#     """The agent class for reinforcement learning

#     The main class that define a model perspective bahaviour of interactions with the environment
#     taking into consideration that we also need to learn from the interactions.
#     """

#     env: EnvironmentDataset
#     """The environment dataset required in RL

#     WARNING: The value of the env is only in the main process and not in the workers
#     """
#     # optimizer: Literal["Adam", "RMSprop", "AdamW"] = "RMSProp"

#     gamma: float = 0.99
#     """Discount factor"""
#     epsilon: HYPERPARAM = Hyperparam(start=0.9, end=0.1)
#     """Epsilon greedy parameter"""
#     batch_size: int = 32
#     """Batch size for learning"""
#     history: int = 4
#     """Number of history to take into consideration"""
#     reward_shaping: bool = True
#     """Whether to shape the rewards"""
#     interactions_per_learn: int = 1
#     """Number of interactions per forward cycle"""
#     policy: Module = None
#     """The policy network for making actions"""

#     queues: Any | RLQueues
#     """Way of communicating with the environment"""

#     _env_info: dict[str, Any] = {}
#     """Information about the environment"""
#     _current_env: Environment = None
#     """The current environment (train or eval env)"""
#     _states = []
#     _lifetime = 0
#     """The lifetime of the agent"""
#     _requires_merge: bool = False
#     _queue_idx: int = 0
#     """The currrent index of the queue the agent is interacting with"""

#     @validator("history")
#     def validate_history(cls, value, values):
#         history = max(value, 1)
#         if len(values["env"].preprocessed_shape) <= 1:
#             history = 1
#         return history

#     @property
#     def queue(self):
#         return (
#             self.queues.train[self._queue_idx]
#             if self.training
#             else self.queues.val_test[self._queue_idx]
#         )

#     @property
#     def active_env(self):
#         return self._current_env

#     @property
#     def is_mp(self):
#         return len(self.queues) > 0

#     @property
#     def view(self):
#         return self.active_env.view

#     @abstractmethod
#     def setup_policy(self) -> nn.Module: ...

#     def init(self):
#         super().init()

#         if self.training:
#             self._current_env = self.env.get_train()
#         else:
#             self._current_env = self.env.get_val()

#         if self.policy is None:
#             self.policy = self.setup_policy()

#     def set_batch_worker_idx(self, batch: dict):
#         if self.is_mp:
#             self._queue_idx = batch["worker"][-1].item()

#     def _get_state(self) -> StateDict:
#         if self.is_mp:
#             try:
#                 state, self._env_info = self.queue.env2agent.get(timeout=2)
#             except Empty as e:
#                 self.error(f"Queue {self._queue_idx} timed out", e)
#                 raise e
#         else:
#             state = self.active_env.last(self.history)
#             self._env_info = dict(memory=len(self.active_env.buffer))
#         return state

#     def _send_act(self, action: np.ndarray, remaining: int):
#         if self.is_mp:
#             self.queue.agent2env.put((action, remaining), timeout=1)
#         else:
#             self.active_env.step(action)

#     def env_info(self):
#         return self._env_info

#     def interact(self, batch: dict):
#         interactions = self.interactions_per_learn if self.training else 1
#         for rem in reversed(range(interactions)):
#             # delayed = list(self.memory.delayed_key_map.keys())
#             state = self._get_state()
#             obs = state["next_obs"]
#             done = bool(state["done"][-1].item())

#             if not self.training and rem == interactions - 1:
#                 assert torch.isclose(batch["obs"][-1].cpu(), state["obs"]).all(), (
#                     "Queue is not in sync for validation"
#                 )

#             # Preprocess the observations
#             obs = self.active_env.preprocess(obs)["image"]

#             # FIXME putting this in the forward might not be the best idea
#             # It will take time to put on the device
#             obs = torch.as_tensor(obs, device=self.device)

#             # In eval, The environment will not ask for an action after being done
#             if self.training or not done:
#                 action, *storables = self.act(obs)
#                 action = action.detach().cpu().numpy().flatten()
#                 # Send the action to the environment
#                 self._send_act(action, remaining=rem)

#             self.step()

#             if not self.training and done:
#                 # We also neeed to break in eval to prevent looping even if we are done
#                 break

#     @property
#     def num_queues(self):
#         if self.is_mp:
#             return len(self.queues)
#         return 0

#     def forward(self, batch):
#         self.set_batch_worker_idx(batch)
#         # First interact with the environment
#         with torch.no_grad():
#             self.interact(batch)
#         # Then learn from the batch
#         return self.learn(batch)

#     def act(self, obs: torch.Tensor, *others: torch.Tensor):
#         other = ()
#         if self.epsilon != 0 and torch.rand(1) < self.epsilon:
#             action = torch.rand(self.active_env.out_action)
#             if len(others) > 0:
#                 elems = [obs.unsqueeze(dim=0)] + [o.unsqueeze(dim=0) for o in others]
#                 _, *other = self.process_act(self.policy(*elems))
#         else:
#             elems = [obs.unsqueeze(dim=0)] + [o.unsqueeze(dim=0) for o in others]
#             action, *other = self.process_act(self.policy(*elems))
#         return (action, *other)

#     def process_act(self, x: torch.Tensor | tuple[torch.Tensor | Any]):
#         if isinstance(x, torch.Tensor):
#             x = [x]
#         if isinstance(x, (list, tuple)):
#             x = [y for y in x if y is not None]
#         data = self.unbatch(x)
#         data[0] = data[0].softmax(dim=-1)
#         return data

#     def unbatch(self, data: Any):
#         if isinstance(data, (list, tuple)):
#             return [self.unbatch(d) for d in data]
#         return data.unsqueeze(dim=0)

#     def last(self, history: int = 0):
#         if self.is_mp:
#             raise NotImplementedError
#         return self.active_env.last(history).to(self.device)

#     def setup_delayed(
#         self, delayed_key_map: dict[str, str], shapes: list[tuple[int, ...]]
#     ):
#         self.active_env.setup_delayed(delayed_key_map, shapes)

#     def state_dict(self, destination=None, prefix="", keep_vars=False):
#         state_dict = super().state_dict(
#             destination=destination, prefix=prefix, keep_vars=keep_vars
#         )
#         state_dict.update({k: self.__getattribute__(k) for k in self._states})
#         return state_dict

#     def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
#         [setattr(self, k, state_dict.pop(k)) for k in self._states if k in state_dict]
#         super().load_state_dict(state_dict, strict)

#     def register_state(self, names: str | list[str]):
#         self._states.extend(names if isinstance(names, (list, tuple)) else [names])

#     def drop(self, env: Environment):
#         if not self.is_mp:
#             self._current_env = env

#     def step(self):
#         self._lifetime += 1
#         if self.training:
#             self.epsilon.step(self._lifetime)

#     def memorize(self, memo: dict):
#         self.active_env.buffer.memorize(memo)

#     def optimize(
#         self,
#         loss: torch.Tensor,
#         optimizer: torch.optim.Optimizer,
#         scheduler: torch.optim.lr_scheduler.LRScheduler = None,
#         clip=None,
#     ):
#         optimizer.zero_grad()
#         loss.backward()
#         if clip is not None:
#             torch.nn.utils.clip_grad.clip_grad_norm_(self.parameters(), clip)
#             # for group in optimizer.param_groups:
#             #     for param in group["params"]:
#             #         if param.grad is not None:
#             #             param.grad.data.clamp_(-clip, clip)
#         optimizer.step()
#         if scheduler is not None:
#             scheduler.step()

#     def train(self, mode: bool = True):
#         if mode:
#             self.epsilon.train_()
#             self.drop(self.env.get_train())
#         else:
#             self.epsilon.eval_()
#             self.drop(self.env.get_val())
#         return super().train(mode)
