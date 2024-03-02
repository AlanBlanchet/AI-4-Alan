from abc import abstractmethod
from typing import Any, Mapping, TypeVar

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from ..env.buffer import DequeueBuffer
from ..env.state import StateDict
from ..utils.hyperparam import Hyperparam, parse_hyperparam
from ..utils.types import STACK_TYPE
from .collect.collector import Collector, DistributedCollector

T = TypeVar("T")


class Policy(nn.Module):
    def __init__(
        self,
        collector: Collector | DistributedCollector,
        stack_type: STACK_TYPE = None,
    ):
        super().__init__()
        self.collector = collector
        self.logger: SummaryWriter = None
        self.in_state = self.collector.state_shape
        self.out_actions = self.collector.out_action
        self.stack_type = stack_type
        self.states = {}

    def setup(self):
        ...

    @abstractmethod
    def act(self, obs: torch.Tensor):
        """
        Base act method for RL algorithms.

        ## Parameters:
        - `obs` (any): The current observation of the environment.

        ## Returns:
        - `action` (any): The action taken in the current state.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Perform batch action selection based on the given states.

        ## Parameters:
        - `obs` (`torch.Tensor`): The observations for which to select actions.

        ## Returns:
        - `actions` (`torch.Tensor`): The selected actions for each observation.
        """
        ...

    @abstractmethod
    def forward(self, batch: StateDict) -> torch.Tensor:
        ...

    @abstractmethod
    def step(self) -> StateDict:
        ...

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def global_step(self):
        return self.collector._n_steps

    def set_logger(self, logger):
        self.logger = logger

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        state_dict.update(self.states)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        [
            setattr(self, k, state_dict.pop(k))
            for k in self.states.keys()
            if k in state_dict
        ]
        super().load_state_dict(state_dict, strict)

    def register_state(self, name: str, variable: T):
        self.states.update({name: variable})
        return variable


class DistributedPolicy(nn.Module):
    def __init__(
        self,
        collector: Collector,
        stack_type: STACK_TYPE = None,
    ):
        super().__init__()
        self.collector = collector
        self.logger: SummaryWriter = None
        self.in_state = self.collector.state_shape
        self.out_actions = self.collector.out_action
        self.stack_type = stack_type
        self.states = {}

    def setup(self):
        ...

    @abstractmethod
    def act(self, obs: torch.Tensor):
        """
        Base act method for RL algorithms.

        ## Parameters:
        - `obs` (any): The current observation of the environment.

        ## Returns:
        - `action` (any): The action taken in the current state.
        """
        raise NotImplementedError

    @abstractmethod
    def batch_act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Perform batch action selection based on the given states.

        ## Parameters:
        - `obs` (`torch.Tensor`): The observations for which to select actions.

        ## Returns:
        - `actions` (`torch.Tensor`): The selected actions for each observation.
        """
        ...

    @abstractmethod
    def forward(self, batch: StateDict) -> torch.Tensor:
        ...

    @abstractmethod
    def step(self) -> StateDict:
        ...

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def global_step(self):
        return self.collector._n_steps

    def set_logger(self, logger):
        self.logger = logger

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        state_dict.update(self.states)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        [
            setattr(self, k, state_dict.pop(k))
            for k in self.states.keys()
            if k in state_dict
        ]
        super().load_state_dict(state_dict, strict)

    def register_state(self, name: str, variable: T):
        self.states.update({name: variable})
        return variable


class EpsilonGreedyLearner(Policy):
    def __init__(
        self,
        epsilon: Hyperparam | int | float | dict,
        learn_steps_counter: int = 0,
    ):
        super().__init__(epsilon, learn_steps_counter)
        self.epsilon = parse_hyperparam(epsilon)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            # Randomly sample from action space for observation/exploitation
            return np.random.choice(self.action_shape)

        # Get the possible actions distribution for the Q matrix
        return np.argmax(self.Q[state])

    def step(self, step=None):
        super().step(step)
        self.epsilon.step(step)

    def reset(self):
        super().reset()
        self.epsilon.reset()


class MemoryLearner(Policy):
    def __init__(
        self,
        in_state: int,
        out_action: int,
        memory_size: int = 10_000,
        train_batch_size: int = 32,
        train_freq: int = 16,
        random_replay: bool = False,
        needed_memory_size: int = 32,
        learn_steps_counter: int = 0,
    ):
        super().__init__(in_state, out_action, learn_steps_counter)
        self.memory_size = memory_size
        self.train_batch_size = train_batch_size
        self.train_freq = train_freq
        self.random_replay = random_replay
        self.needed_memory_size = needed_memory_size
        self.memory = DequeueBuffer(size=self.memory_size)
        self.replays = 0

    def learn(self, state, action, r, state_next, done, *args):
        """
        Learn from an experience tuple.

        ## Parameters:
        - `state` (any): The current state of the environment.
        - `action` (any): The action taken in the current state.
        - `r` (float): The reward received after taking the action.
        - `state_next` (any): The next state of the environment.
        - `done` (bool): A flag indicating whether the episode is done.
        - `*args` (any): Additional arguments to pass to the `replay` method.

        ## Returns:
        None

        ## Notes:
        - This method can be overridden to implement additional an learning algorithm.
        - The `replay` method should be overridden to define how the learning algorithm replays experiences.
        """
        self.memory.push(state, action, r, state_next, done, *args)

        sample = None
        if self.can_replay(done):
            self.replays += 1
            self.logger.add_scalar("stats/replay_count", self.replays, self.global_step)

            if self.random_replay:
                sample = self.memory.sample(self.train_batch_size)
            else:
                sample = self.memory[-self.train_batch_size :]

            sample = [
                self._parse_tensor(x).to(self.device) for x in sample
            ]  # list[M, B]
            self.replay(sample)

    def can_replay(self, _: bool):
        """
        Check if the agent can replay a memory batch for training.

        ## Parameters
        - `done` (bool): Indicates whether the current episode is done.

        ## Returns
        - `bool`: Returns `True` if the agent can replay a memory batch for training, `False` otherwise.

        The agent can replay a memory batch for training if the following conditions are met:
        - The number of learn steps is greater than or equal to `self.needed_memory_size`.
        - The current episode is done and `self.train_when_done` is `True`.
        - The number of learn steps is a multiple of `self.train_freq`.

        Example usage:
        ```python
        if agent.can_replay(done=True):
            replay_train(...)
        ```
        """

        if self.learn_steps_counter < self.needed_memory_size:
            # Memory needs to be larger, no training yet
            return False

        if self.learn_steps_counter % self.train_freq != 0:
            # Update parameters every `self.train_freq` steps
            # This way we add inertia to the agent actions, as they are more sticky
            return False

        return True

    @abstractmethod
    def replay(self, batch: list[torch.Tensor]):
        """
        Replays a batch of experiences and updates algorithms.

        ## Parameters:
        - `batch` (list[torch.Tensor]): A list of experiences to replay.

        ## Returns:
        None

        ## Raises:
        - `NotImplementedError`: If the method is not implemented in a derived class.
        """
        raise NotImplementedError()

    def _parse_tensor(self, data, dtype=torch.float32):
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(dtype)
        elif isinstance(data, torch.Tensor):
            return data.clone().detach()
        elif isinstance(data, (list, tuple)):
            if isinstance(data[0], np.ndarray):
                return torch.stack([torch.from_numpy(d) for d in data]).to(dtype)
            elif isinstance(data[0], torch.Tensor):
                return torch.stack(data).to(dtype)
        return torch.tensor(data, dtype=dtype)

    def reset(self):
        super().reset()
        self.logger.add_scalar("stats/memory_size", len(self.memory), self.global_step)
        self.memory.reset()
        self.memory_sample_count = 0


class TrajectoryLearner(MemoryLearner):
    def __init__(self, in_state: int, out_action: int, replay_size=1024):
        super().__init__(
            in_state,
            out_action,
            memory_size=replay_size,
            train_batch_size=replay_size,
            train_freq=1,
            needed_memory_size=1,
        )

    @abstractmethod
    def replay(self, batch: list[torch.Tensor]):
        """
        Replays a batch of experiences from a single trajectory and updates algorithms.

        ## Parameters:
        - `batch` (list[torch.Tensor]): A list of experiences to replay.

        ## Returns:
        None

        ## Raises:
        - `NotImplementedError`: If the method is not implemented in a derived class.
        """
        raise NotImplementedError()

    def can_replay(self, done: bool):
        return done
