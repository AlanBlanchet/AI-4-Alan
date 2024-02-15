import gymnasium
import torch
from gymnasium.spaces import Discrete

from .buffer import SmartReplayBuffer
from .utils import get_env_config, get_preprocess


class Environment:
    def __init__(self, env_name: str):
        self.env_name = env_name
        self._env = gymnasium.make(self.env_name, render_mode="rgb_array")
        self.memory = SmartReplayBuffer(4_000, self.observation_shape, self.out_action)
        self.preprocess_fn = get_preprocess(env_name)
        self._current_obs = None
        self.reset()

    def reset(self):
        self._current_obs, _ = self._env.reset()
        return self._current_obs

    def close(self):
        self.reset()
        self._env.close()

    def step(self, action: torch.Tensor):
        action = action.detach().cpu().numpy()
        discrete_action = action.argmax().item()
        obs = self._current_obs
        next_obs, reward, terminate, truncated, _ = self._env.step(discrete_action)
        done = terminate or truncated
        self._current_obs = next_obs
        experience = (obs, action, reward, next_obs, done)
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

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        del state["_env"]
        return state

    def clone(self):
        return Environment(self.env_name)
