import gymnasium
import torch


class Environment:
    def __init__(self, env_id: str):
        self._env = gymnasium.make(env_id, render_mode="rgb_array")

    def reset(self):
        state = self._env.reset()
        return torch.from_numpy(state).float()

    def step(self, action):
        return self._env.step(action)

    def set_render(self, render_mode: str):
        self._env.close()
        self._env = gymnasium.make(self._env.spec.id, render_mode=render_mode)
        return self

    @property
    def render_mode(self):
        return self._env.render_mode
