import ale_py  # noqa
import gymnasium
import torch


class AdaptedEnv:
    def __init__(self, name: str, seed=0):
        self.name = name
        self.seed = seed
        self.env = gymnasium.make(name, render_mode="rgb_array")
        self._seeded = False

    @property
    def view(self):
        return self.env.render()

    @property
    def render_mode(self):
        return self.env.render_mode

    @property
    def simulated(self):
        return len(self.observation_space.shape) != 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        if self._seeded:
            return self.env.reset()
        self._seeded = True
        return self.env.reset(seed=self.seed)

    def close(self):
        return self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def random_action(self):
        return torch.tensor(self.env.action_space.sample())

    @property
    def unwrapped(self):
        return self.env.unwrapped
