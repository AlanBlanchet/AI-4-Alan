# TODO fix version
from __future__ import annotations

import queue
from functools import cache, cached_property
from multiprocessing import Queue
from pathlib import Path
from threading import Thread

import numpy as np
import torch
from gymnasium.spaces import Discrete
from torch.utils.data import Dataset, IterableDataset

from ...configs.base import BaseMP
from ...configs.log import Color
from ...utils.func import TensorInfo
from ..base_dataset import BaseDataset
from .adapter import AdaptedEnv
from .buffer import ReplayBuffer
from .queues import SplitQueue
from .state import StateDict
from .utils import get_env_config, get_preprocess
from .video import VideoManager


class Environment(BaseMP):
    """The environment class that wraps the gym environment

    Instances can be used with multiple workers in a fork manner
    """

    model_config = {"arbitrary_types_allowed": True}

    log_name = "env"
    color = Color.lime

    name: str
    memory: int
    history: int
    preprocess_buffer: bool = True

    required_batch_size: int = 1
    total_steps: int = -1
    max_steps: int = -1

    skips: int = 0
    timeout: int = 5

    _last_action: int = None
    _steps_without_different_action: int = 0
    _log_p: Path = None
    _video_manager: VideoManager = None
    _lifetime: int = 0
    _requested_items: int = 0
    _env: AdaptedEnv = None
    _buffer: ReplayBuffer = None
    _agent2env: Queue = None
    _env2agent: Queue = None
    _last_transition: tuple = None

    @classmethod
    def log_extras(cls):
        txt = super().log_extras()
        if cls.worker_id is not None:
            return f"{txt}[{cls.__name__}]"
        return txt

    @property
    def is_val(self):
        return self.total_steps < 0

    @property
    def current_obs(self):
        return self.last_delayed["next_obs"].to(torch.float32)

    @property
    def last_delayed(self):
        return self.buffer._last_delayed

    @property
    def buffer(self):
        return self._buffer

    @property
    def _is_over_max_steps(self):
        return (
            self._steps_without_different_action >= self.max_steps
            and self.max_steps > 0
        )

    @cached_property
    def preprocess_fn(self) -> str:
        return get_preprocess(self.name)

    @property
    def render_mode(self):
        return self._env.render_mode

    @property
    def observation_shape(self):
        if isinstance(self._env.observation_space, Discrete):
            return self._env.observation_space.n
        else:
            return self._env.observation_space.shape

    @property
    def preprocessed_shape(self):
        config = get_env_config(self.name)
        return config.get("out_shape", self.observation_shape)

    @property
    def effective_shape(self):
        return (
            self.preprocessed_shape
            if self.preprocess_buffer
            else self.observation_shape
        )

    @property
    def out_action(self):
        if isinstance(self._env.action_space, Discrete):
            return self._env.action_space.n
        else:
            return self._env.action_space.shape

    @property
    def action_names(self):
        unwrapped = self._env.unwrapped
        if hasattr(unwrapped, "get_action_meanings"):
            return unwrapped.get_action_meanings()
        return range(self.out_action)

    @property
    def config(self):
        return get_env_config(self.name)

    @property
    def view(self):
        return self._env.view

    @property
    def simulated(self):
        return self._env.simulated

    @property
    def is_first_process(self):
        """Check if the current process is the first process.

        First refers to the main process in a forked environment or the only process in a single environment.
        """
        return self.worker_id is None or self.worker_id == 0

    @property
    def done(self):
        if self._last_transition is None:
            return False
        return self._last_transition[-2]

    def model_post_init(self, __context):
        super().model_post_init(__context)

        self.init()

    def init(self, seed=None):
        self._env = AdaptedEnv(name=self.name, seed=seed)
        self._buffer = ReplayBuffer(
            capacity=self.memory,
            obs_info=TensorInfo(shape=self.effective_shape, dtype=torch.uint8),
            action_info=TensorInfo(shape=self.out_action, dtype=torch.float32),
        )
        self.reinitialize()

        name = self.name if self.worker_id is None else f"{self.name}_{self.worker_id}"
        interaction_thread = Thread(target=self.interaction_loop, args=(), name=name)
        interaction_thread.start()

    def interaction_loop(self):
        while True:
            self._poll()

    @BaseMP.watch
    def reinitialize(self, send=True):
        self.info("Reinitializing environment")

        self._lifetime = 0
        self._requested_items = 0
        self.buffer.clear()
        self.reset()

        if self.is_mp:
            # Check if the queues are empty
            agent2env = self._agent2env.qsize()
            split = "Val" if self.is_val else "Train"
            msg = f"{split} agent2env queue should be empty when re-initializing the environment"
            assert agent2env == 0, f"{msg} - {agent2env=}"

        self.empty()

        # The agent requires to read a first observation at the start !
        self.burn(1, send=send)

    def empty(self):
        if self.is_mp:
            while not self._env2agent.empty():
                self._env2agent.get()
            while not self._agent2env.empty():
                self._agent2env.get()

    def init_worker(self, worker_id, queues: SplitQueue):
        """Initialize the environment for a worker

        Writing to a variable creates a new memory space for the variable
        Reading from a variable reads from the shared memory space with the parent process
        """
        self._agent2env = queues.agent2env
        self._env2agent = queues.env2agent

        self.info(
            f"Initializing {self.__class__.__name__} worker in forked environment"
        )
        self.init(seed=worker_id)
        self.reinitialize(send=False)

        # Initialize default variables for each worker
        self._log_p = self._log_p / f"worker_{worker_id}"
        self._log_p.mkdir(exist_ok=True, parents=True)
        self.setup_video()

        # Workers get their own memory space (except for the 0 ?)

        # Let the iterator send to the queue
        # self.reinitialize(send=False)

    @BaseMP.watch
    def setup_video(self):
        if self.is_val:
            path_name = "val"
            shape = self.view.shape[:-1][::-1]
        else:
            path_name = "train"
            shape = self.effective_shape[1:]

        self._video_manager = VideoManager(
            log_path=self._log_p / path_name,
            shape=shape,
            stream=not self.is_val,
            open_stream=self.is_first_process,
        )

    def set_run_p(self, run_p: Path):
        self._log_p = run_p / "video"
        self._log_p.mkdir(exist_ok=True, parents=True)
        # self.setup_video()

    def reset(self):
        self.debug("Environment reset called")
        next_obs, _ = self._env.reset()
        self.buffer._last_delayed["next_obs"] = self._preprocess(next_obs)
        self._steps_without_different_action = 0

        if self.is_val and self._video_manager:
            self._video_manager.create_file()
            self._video_manager.write(self.view)

        return self.current_obs

    def setup_delayed(
        self, delayed_key_map: dict[str, str], shapes: list[tuple[int, ...]]
    ):
        self.buffer.setup_delayed(delayed_key_map, shapes)

    def close(self):
        self.reset()
        self._env.close()

    def log_transition(self, next_obs: torch.Tensor = None):
        if self._video_manager:
            if self.is_val:
                self._video_manager.write(self.view)
                if self.done:
                    self.debug(f"Saving video to {self._log_p}")
                    self._video_manager.save()
            elif next_obs is not None:
                # train obs
                self._video_manager.write(
                    next_obs.permute(1, 2, 0).repeat(1, 1, 3).numpy()
                )

    @BaseMP.watch
    def step(self, action: np.ndarray, *storables: np.ndarray, send: bool = False):
        discrete_action = action.argmax().item()

        # Frame skip
        done_while_skipping = False
        for _ in range(self.skips):
            # The env can terminate while we are skipping frames !
            # If we don't catch it, we will be stuck (missed a done flag)
            _, _, terminate, truncated, _ = self._env.step(discrete_action)
            done_while_skipping = terminate or truncated or done_while_skipping
            if self.is_val:
                # We only show the real environment during val
                self.log_transition()

        obs, *delayed = self.last_delayed.values()
        next_obs, reward, terminate, truncated, _ = self._env.step(discrete_action)
        done = terminate or truncated or done_while_skipping or self._is_over_max_steps
        # INFO debugging
        if (
            self.is_val
            and self._requested_items == 30
            and self.is_mp
            and self.worker_id == 0
        ):
            done = True

        next_obs = self._preprocess(next_obs)
        transition = (obs, action, reward, next_obs, done, (delayed, storables))
        self._last_transition = transition
        self.buffer.store(transition)
        self._lifetime += 1

        self.log_transition(next_obs)

        # Prevent going in a while loop
        if self._last_action != discrete_action:
            self._last_action = discrete_action
            self._steps_without_different_action = 0
        else:
            self._steps_without_different_action += 1

        if self._lifetime % 100 == 0:
            self.debug(f"Environment total steps {self._lifetime}")

        if self.is_mp and send:
            out = self.buffer.last(self.history).format_out()
            self.info("Sending", out["obs"].mean())
            self._env2agent.put(
                (dict(out), dict(memory=len(self.buffer))), timeout=self.timeout
            )

        if done:
            self.reset()

        return transition

    def burn(self, n: int, send=False):
        self.debug(f"Burning {n} steps")
        for _ in range(n):
            action = self._env.random_action()
            self.step(action, send=send)

    def _preprocess(self, obs):
        if self.preprocess_buffer:
            return self.preprocess_fn(obs)
        return obs

    def preprocess(self, obs):
        if not self.preprocess_buffer:
            return self.preprocess_fn(obs)
        return obs

    def add_worker_info(self, out: StateDict):
        if self.is_mp:
            out["worker"] = self.worker_id
        return out

    def sample(self):
        """
        Use a strategy to sample the states from the buffer
        """
        return self.add_worker_info(
            self.buffer.sample(1, self.history).format_out().squeeze(0)
        )

    def last(self, n: int):
        return self.add_worker_info(self.buffer.last(n).format_out())

    def _check(self):
        stop_next_iter = False

        if self._requested_items == 0:
            self.reinitialize(send=True)

        if not self.is_mp and self.is_val and self.done:
            stop_next_iter = True

        if (
            self.is_mp
            and self._requested_items % self.required_batch_size == 0
            and self._requested_items != 0
        ):
            try:
                remaining = 1
                while remaining > 0:
                    self.info("Waiting for action")
                    action, remaining = self._agent2env.get(timeout=self.timeout)
                    self.info(f"Got {action=}, {remaining=}")
                    transition = self.step(action, send=True)
                    stop_next_iter |= bool(transition[-2])
                    self.info(f"Transition {transition[-2]}")

                    if self.is_val and stop_next_iter:
                        # Stop the next iteration if the environment is done and in val
                        break
            except queue.Empty as e:
                self.error(
                    "Queue is empty.",
                    self,
                    f" {self._requested_items=}, {self.done=}, {self._lifetime=}, {self._steps_without_different_action=}",
                )
                raise e

        self._requested_items += 1
        self.info(f"Requested {self._requested_items} items. Done ? {stop_next_iter}")
        return stop_next_iter

    def _poll(self): ...


class ValEnvironment(Environment, IterableDataset):
    _stop_next_iter: bool = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._stop_next_iter:
            self._stop_next_iter = False
            self.warn(
                f"Stopped at lifetime {self._lifetime}, requested {self._requested_items} items"
            )
            # Next turn will call the reinitialize
            self._requested_items = 0
            raise StopIteration

        self._stop_next_iter = self._poll()

        return self.last(self.history)


class TrainEnvironment(Environment, Dataset):
    def __getitem__(self, _):
        """We don't care about the index since we manually sample the data"""
        self._poll()
        return self.sample()

    def __len__(self):
        return self.total_steps * self.required_batch_size


class EnvironmentDataset(BaseDataset):
    """Environment that acts as a dataset

    This dataset constructs the environments (train and val) and provides the data to the agent
    """

    model_config = {"arbitrary_types_allowed": True}

    memory: int = 10_000
    """Size of the replay buffer"""
    history: int = 4
    """Number of frames to stack"""
    skips: int = 0
    """Also do frame skipping during val"""
    skips_in_val: bool = False
    """Number of frames to skip"""
    preprocess_buffer: bool = True
    """Whether to preprocess the buffer or not before storing it"""
    batch_size: int = 1
    """Batch size for the dataloader"""
    steps: int = -1
    """Number of steps in the environment per epoch"""

    @classmethod
    def get_identifiers(cls):
        return super().get_identifiers() | {"gym", "gymnasium"}

    def post_config(self, config: dict):
        super().post_config(config)

        self.batch_size = config["task"]["datamodule"]["batch_size"]

    def prepare(self, run_p: Path):
        train = self.get_train()
        val = self.get_val()

        self.info(f"Setting run path to {run_p}")
        train.set_run_p(run_p)
        val.set_run_p(run_p)

    def create_env(self, total_steps: int = -1) -> Environment:
        is_val = total_steps < 0
        env_cls = ValEnvironment if is_val else TrainEnvironment

        extra = (
            {"total_steps": total_steps, "required_batch_size": self.batch_size}
            if not is_val
            else {}
        )

        return env_cls(
            name=self.name,
            memory=self.history if is_val else self.memory,
            preprocess_buffer=self.preprocess_buffer,
            max_steps=self.max_steps,
            skips=0 if is_val and not self.skips_in_val else self.skips,
            history=self.history,
            **extra,
        )

    @cache
    def get_train(self) -> Environment:
        return self.create_env(self.steps)

    @cache
    def get_val(self) -> Environment:
        return self.create_env()

    def __hash__(self):
        # TODO Check if this is correct
        return hash(self.name)

    @cached_property
    def name(self) -> str:
        return self.params["name"]

    @cached_property
    def max_steps(self):
        return self.params.get("max_steps", -1)

    def parse_items(self, item, map):
        return item

    def item_from_id(self, id, split):
        raise NotImplementedError()

    def extract_inputs(self, item: dict) -> dict:
        return item
