# TODO fix version
from functools import cache, cached_property
from multiprocessing import Queue
from pathlib import Path

import numpy as np
from gymnasium.spaces import Discrete
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ...configs.base import Base
from ...configs.log import Color
from ..base_dataset import BaseDataset
from .adapter import AdaptedEnv
from .buffer import SmartReplayBuffer
from .utils import get_env_config, get_preprocess
from .video import VideoManager


class Environment(Base):
    """The environment class that wraps the gym environment

    Instances can be used with multiple workers in a fork manner
    """

    log_name = "env"
    color = Color.lime

    class Config:
        arbitrary_types_allowed = True

    name: str
    memory: int = 10_000
    preprocess_buffer: bool = True

    steps: int = -1
    max_steps: int = -1

    steps_without_different_action: int = 0
    last_action: int = None

    skips: int = 0

    _log_p: Path = None
    _video_manager: VideoManager = None
    _batch_size: int = 1
    _lifetime: int = 0
    _requested_items: int = 0
    _env: AdaptedEnv = None
    _buffer: SmartReplayBuffer = None
    _agent2env: Queue = None
    _env2agent: Queue = None
    _done: bool = False

    @classmethod
    def log_extras(cls):
        """Manages logging for a single instance or a worker instance"""
        txt = super().log_extras()
        if worker_info := get_worker_info():
            return txt + f"[{cls.__name__} Worker°{worker_info.id}] "
        return txt

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.init()

    @property
    def is_val(self):
        return self.steps < 0

    @property
    def current_obs(self):
        return self.buffer.next_elements["next_obs"]

    @property
    def delayed_elements(self):
        return self.buffer.next_elements

    @property
    def buffer(self):
        return self._buffer

    @property
    def _is_over_max_steps(self):
        return (
            self.steps_without_different_action >= self.max_steps and self.max_steps > 0
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

    def init(self, seed=None):
        self._env = AdaptedEnv(name=self.name, seed=seed)
        self._buffer = SmartReplayBuffer(
            self.memory, self.effective_shape, self.out_action
        )

        self.reinitialize()

    def reinitialize(self):
        self._lifetime = 0
        self._requested_items = 0
        self.buffer.clear()
        self.reset()
        self.burn(1)

    def init_worker(self, worker_id, agent2env: Queue, env2agent: Queue):
        """Initialize the environment for a worker

        Writing to a variable creates a new memory space for the variable
        Reading from a variable reads from the shared memory space with the parent process
        """
        if self._agent2env is None:
            # Initialize default variables for each worker
            self._agent2env = agent2env
            self._env2agent = env2agent
            self._log_p = self._log_p / f"worker_{worker_id}"
            self._log_p.mkdir(exist_ok=True, parents=True)
            self._video_manager = VideoManager(
                log_path=self._log_p, shape=self.view.shape[:-1][::-1]
            )

            # TODO fix this. We need to do something for the memory space in the main process !
            # A main environment exists but is never used !
            if worker_id is None:  # TODO change this condition
                # Worker 0 uses variables from the parent process
                self.info("Initializing shared environment with main process")
                self.reinitialize()
            else:
                # Other workers get their own memory space
                self.info("Initializing forked environment in worker")
                self.init(seed=worker_id)

    def clear_buffer(self):
        self._buffer.clear()

    def set_run_p(self, run_p: Path):
        self._log_p = run_p / "video"
        self._log_p.mkdir(exist_ok=True, parents=True)

    def set_batch_size(self, batch_size: int):
        self._batch_size = batch_size

    def reset(self):
        self.debug(f"Environment ({id(self)}) reset called")
        next_obs, _ = self._env.reset()
        self.buffer.next_elements["next_obs"] = self._preprocess(next_obs)
        self.steps_without_different_action = 0

        if self.is_val and self._video_manager:
            self._video_manager.create_file()
            self._video_manager.write(self.view)

        if self._env2agent:
            self._env2agent.put_nowait((self.buffer.last(4), dict()))

        return self.current_obs

    def setup_delayed(
        self, delayed_key_map: dict[str, str], shapes: list[tuple[int, ...]]
    ):
        self.buffer.setup_delayed(delayed_key_map, shapes)

    def close(self):
        self.reset()
        self._env.close()

    def log_experience(self):
        if self.is_val and self._video_manager:
            self._video_manager.write(self.view)
            if self._done:
                self.debug(f"Saving video to {self._log_p}")
                self._video_manager.save()

    def step(self, action: np.ndarray, *storables: np.ndarray):
        discrete_action = action.argmax().item()
        obs, *delayed = self.delayed_elements.values()
        next_obs, reward, terminate, truncated, _ = self._env.step(discrete_action)
        self._done = terminate or truncated or self._is_over_max_steps
        next_obs = self._preprocess(next_obs)
        experience = (obs, action, reward, next_obs, self._done, (delayed, storables))
        self.buffer.store(experience)
        self._lifetime += 1

        self.log_experience()

        # Prevent going in a while loop
        if self.last_action != discrete_action:
            self.last_action = discrete_action
            self.steps_without_different_action = 0
        else:
            self.steps_without_different_action += 1

        if self._lifetime % 100 == 0:
            self.debug(f"Environment total steps {self._lifetime}")

        # Frame skip
        for _ in range(self.skips):
            self._env.step(discrete_action)

        if self._env2agent:
            self._env2agent.put_nowait(
                (self.buffer.last(4), dict(memory=len(self.buffer)))
            )

        if self._done:
            self.reset()

        return experience

    def burn(self, n: int):
        self.debug(f"Burning {n} steps")
        for _ in range(n):
            action = self._env.random_action()
            self.step(action)

    def _preprocess(self, obs):
        if self.preprocess_buffer:
            return self.preprocess_fn(obs)
        return obs

    def preprocess(self, obs):
        if not self.preprocess_buffer:
            return self.preprocess_fn(obs)
        return obs

    def sample(self):
        """
        Use a strategy to sample the states from the buffer
        """
        return self.buffer.sample(1, 4).squeeze(0)

    def check_queues(self):
        if self._agent2env:
            action = self._agent2env.get()
            # if action:
            self.step(action)

    # def clone(self, memory=1):
    #     env = Environment(self.name, memory, False)
    #     delayed_key_maps = {**self.buffer.delayed_key_map}
    #     delayed_key_maps.pop("next_obs")
    #     env.setup_delayed(delayed_key_maps, self.buffer.delayed_shapes)
    #     return env


class IterableEnvironment(Environment, IterableDataset):
    def __iter__(self):
        # self.reset()
        return self

    def __next__(self):
        state = self.buffer.last(4)

        if self._done:
            self._done = False
            raise StopIteration

        if self._requested_items % self._batch_size == 0 and self._requested_items != 0:
            self.check_queues()

        self._requested_items += 1
        return state


class FixedLenEnvironment(Environment, Dataset):
    def __getitem__(self, _):
        if self._requested_items % self._batch_size == 0 and self._requested_items != 0:
            self.check_queues()

        self._requested_items += 1
        return self.sample()

    def __len__(self):
        return self.steps


class EnvironmentDataset(BaseDataset):
    class Config:
        arbitrary_types_allowed = True

    memory: int = 10_000
    skips: int = 0
    preprocess_buffer: bool = True

    @classmethod
    def get_identifiers(cls):
        return super().get_identifiers() | {"gym", "gymnasium"}

    def prepare(self, run_p: Path, batch_size: int, val_batch_size: int):
        train = self.get_train()
        val = self.get_val()

        self.info(f"Setting run path to {run_p}")
        train.set_run_p(run_p)
        val.set_run_p(run_p)
        train.set_batch_size(batch_size)
        val.set_batch_size(val_batch_size)

    def create_env(self, steps: int = -1) -> Environment:
        is_iter = steps == -1
        env_cls = IterableEnvironment if is_iter else FixedLenEnvironment

        extra = {"steps": steps} if not is_iter else {}

        return env_cls(
            name=self.name,
            memory=self.memory,
            preprocess_buffer=self.preprocess_buffer,
            max_steps=self.max_steps,
            skips=self.skips,
            **extra,
        )

    @cache
    def get_train(self) -> Environment:
        return self.create_env(self.steps_per_epoch)

    @cache
    def get_val(self) -> Environment:
        return self.create_env()

    def __hash__(self):
        # TODO Check if this is correct
        return hash(self.name)

    @cached_property
    def steps_per_epoch(self):
        return self.params.get("steps", -1)

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
