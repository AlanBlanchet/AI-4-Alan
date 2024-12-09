# TODO fix version
import os
from functools import cache, cached_property
from pathlib import Path

import cv2
import torch
from gymnasium.spaces import Discrete
from pydantic import BaseModel
from torch.utils.data import Dataset, IterableDataset

from ..base_dataset import BaseDataset
from .adapter import AdaptedEnv
from .buffer import SmartReplayBuffer
from .utils import get_env_config, get_preprocess


class Environment(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str
    memory: int = 10_000
    preprocess_buffer: bool = True

    steps: int = -1
    max_steps: int = -1

    steps_without_different_action: int = 0
    last_action: int = None
    writer_info: tuple[cv2.VideoWriter, Path] = None

    skips: int = 0

    def model_post_init(self, _):
        self.reset()

        # Make sure we have at least an element for the dataloader checking
        self.burn(1)

    @cached_property
    def preprocess_fn(self) -> str:
        return get_preprocess(self.name)

    @cached_property
    def buffer(self):
        return SmartReplayBuffer(self.memory, self.effective_shape, self.out_action)

    @cached_property
    def env(self):
        return AdaptedEnv(self.name)

    @property
    def is_val(self):
        return self.steps < 0

    def reset(self):
        next_obs, _ = self.env.reset()
        self.buffer.next_elements["next_obs"] = self._preprocess(next_obs)
        self.steps_without_different_action = 0
        if self.is_val:
            self.writer_info = self.get_writer_info()
        return self.current_obs

    @property
    def current_obs(self):
        return self.buffer.next_elements["next_obs"]

    @property
    def delayed_elements(self):
        return self.buffer.next_elements

    def setup_delayed(
        self, delayed_key_map: dict[str, str], shapes: list[tuple[int, ...]]
    ):
        self.buffer.setup_delayed(delayed_key_map, shapes)

    def close(self):
        self.reset()
        self.env.close()

    @property
    def _is_over_max_steps(self):
        return (
            self.steps_without_different_action >= self.max_steps and self.max_steps > 0
        )

    def get_writer_info(self):
        test_p = Path("videos")
        test_p.mkdir(exist_ok=True)

        i = 1
        while (test_p / f"eval_{i}.mp4").exists():
            i += 1

        video_p = test_p / f"eval_{i}.avi"
        writer = cv2.VideoWriter(
            str(video_p),
            cv2.VideoWriter_fourcc(*"MPEG"),
            30,
            self.view.shape[:-1][::-1],
        )
        writer.write(self.view)
        return writer, video_p

    def log_experience(self, experience):
        if self.is_val:
            done = experience[-2]

            writer, path = self.writer_info

            writer.write(self.view)

            if done:
                # Can't directly use the h264 codec for vscode
                writer.release()
                os.system(
                    f"ffmpeg -hide_banner -i {str(path)} -c:v libx264 {path.with_suffix('.mp4')} 2>/dev/null"
                )
                os.remove(path)

    def step(self, action: torch.Tensor, *storables: torch.Tensor):
        action = action.detach().cpu().numpy()
        discrete_action = action.argmax().item()
        obs, *delayed = self.delayed_elements.values()
        next_obs, reward, terminate, truncated, _ = self.env.step(discrete_action)
        done = terminate or truncated or self._is_over_max_steps
        next_obs = self._preprocess(next_obs)
        experience = (obs, action, reward, next_obs, done, (delayed, storables))
        self.buffer.store(experience)

        if done:
            self.reset()

        if self.is_val:
            self.log_experience(experience)

        # Prevent going in a while loop
        if self.last_action != discrete_action:
            self.last_action = discrete_action
            self.steps_without_different_action = 0
        else:
            self.steps_without_different_action += 1

        # Frame skip
        for _ in range(self.skips):
            self.env.step(discrete_action)

        return experience

    def burn(self, n: int):
        for _ in range(n):
            action = self.env.random_action()
            self.step(action)

    @property
    def render_mode(self):
        return self.env.render_mode

    @property
    def observation_shape(self):
        if isinstance(self.env.observation_space, Discrete):
            return self.env.observation_space.n
        else:
            return self.env.observation_space.shape

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

    @property
    def out_action(self):
        if isinstance(self.env.action_space, Discrete):
            return self.env.action_space.n
        else:
            return self.env.action_space.shape

    @property
    def action_names(self):
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "get_action_meanings"):
            return unwrapped.get_action_meanings()
        return range(self.out_action)

    @property
    def config(self):
        return get_env_config(self.name)

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        del state["env"]
        return state

    @property
    def view(self):
        return self.env.view

    @property
    def simulated(self):
        return self.env.simulated

    def clone(self, memory=1):
        env = Environment(self.name, memory, False)
        delayed_key_maps = {**self.buffer.delayed_key_map}
        delayed_key_maps.pop("next_obs")
        env.setup_delayed(delayed_key_maps, self.buffer.delayed_shapes)
        return env


class IterableEnvironment(Environment, IterableDataset):
    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        state = self.buffer.last(4)
        done = state["done"][-1]
        if done:
            raise StopIteration
        return state


class FixedLenEnvironment(Environment, Dataset):
    def __getitem__(self, _):
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
    def create_train(self) -> Environment:
        return self.create_env(self.steps_per_epoch)

    @cache
    def create_val(self) -> Environment:
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
