import gymnasium
import torch
from gymnasium.spaces import Discrete

from .buffer import SmartReplayBuffer
from .utils import get_env_config, get_preprocess


class Environment:
    def __init__(
        self, env_name: str, memory: int = 10_000, preprocess_buffer=True, **_
    ):
        self.env_name = env_name
        self.original = preprocess_buffer
        self.preprocess_fn = get_preprocess(env_name)
        self._env = gymnasium.make(self.env_name, render_mode="rgb_array")
        self.memory = SmartReplayBuffer(memory, self.effective_shape, self.out_action)
        self.reset()

    def reset(self):
        next_obs, _ = self._env.reset()
        self.memory.next_elements["next_obs"] = self._preprocess(next_obs)
        return self.current_obs

    @property
    def current_obs(self):
        return self.memory.next_elements["next_obs"]

    @property
    def delayed_elements(self):
        return self.memory.next_elements

    def setup_delayed(
        self, delayed_key_map: dict[str, str], shapes: list[tuple[int, ...]]
    ):
        self.memory.setup_delayed(delayed_key_map, shapes)

    def close(self):
        self.reset()
        self._env.close()

    def step(self, action: torch.Tensor, *storables: torch.Tensor):
        action = action.detach().cpu().numpy()
        discrete_action = action.argmax().item()
        obs, *delayed = self.delayed_elements.values()
        next_obs, reward, terminate, truncated, _ = self._env.step(discrete_action)
        done = terminate or truncated
        next_obs = self._preprocess(next_obs)
        experience = (obs, action, reward, next_obs, done, (delayed, storables))
        self.memory.store(experience)
        if done:
            self.reset()
        return experience

    def set_render(self, render_mode: str):
        self.close()
        self._env = gymnasium.make(self._env.spec.id, render_mode=render_mode)
        return self

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
        config = get_env_config(self.env_name)
        if config is not None:
            return config["out_shape"]
        return self.observation_shape

    @property
    def effective_shape(self):
        return self.preprocessed_shape if self.original else self.observation_shape

    def _preprocess(self, obs):
        if self.original:
            return self.preprocess_fn(obs)
        return obs

    def preprocess(self, obs):
        if not self.original:
            return self.preprocess_fn(obs)
        return obs

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
        return get_env_config(self.env_name)

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        del state["_env"]
        return state

    def clone(self, memory=1, preprocess=False):
        env = Environment(self.env_name, memory, preprocess)
        delayed_key_maps = {**self.memory.delayed_key_map}
        delayed_key_maps.pop("next_obs")
        env.setup_delayed(delayed_key_maps, self.memory.delayed_shapes)
        return env
