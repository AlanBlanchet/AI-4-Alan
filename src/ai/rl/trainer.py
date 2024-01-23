import json
from pathlib import Path
from time import time
from typing import Literal

import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from ai.utils.paths import AIPaths

from .policy import Policy
from .utils.func import random_run_name


class Trainer:
    """
    Collects data from the environment and trains the agent.
    """

    def __init__(
        self,
        agent: Policy,
        tqdm_reward_update_s: float = 0.2,
        reward_svg_size: int = 50,
        reward_mode: Literal["accumulate", "mean"] = "mean",
    ):
        self.agent = agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tqdm_reward_update_s = tqdm_reward_update_s
        self.reward_svg_size = reward_svg_size
        self.rewards_ = []
        self.train_mode_ = None
        self.run_name = random_run_name()
        self.tensorboard = SummaryWriter(AIPaths.tensorboard / self.run_name)
        self.agent.set_logger(self.tensorboard)
        self.collector.set_logger(self.tensorboard)
        self.reward_mode = reward_mode

        torch.autograd.set_detect_anomaly(True)

    @property
    def collector(self):
        return self.agent.collector

    def train(self, collection_steps=4):
        # Make sure we are in a train render mode
        if self.train_mode_ is None:
            self.train_mode_ = self.collector.render_mode
        if self.train_mode_ != self.collector.render_mode:
            self.collector.set_render(self.train_mode_)

        self.agent.to(self.device)
        self.agent.train()

        return self._run(collection_steps, "train")

    def eval(self, collection_steps=4):
        self.agent.eval()
        return self._run(collection_steps, "eval")

    def test(self, episodes=4):
        self.agent.eval()
        self.collector.set_render("human")

        episode = 0
        while episode < episodes:
            _, __, reward, ___, done = self.collector.collect_step(
                self.agent, store=False
            )

            if done:
                episode += 1
                # self.rewards_.append(reward)
        self.collector.clean()

    def _run(self, collection_steps, mode="train"):
        self._reset_stats()

        start_time = time()
        delta_time = 0

        pbar = tqdm(range(int(collection_steps)))

        for step in pbar:
            delta_time = time() - start_time

            env_data = self.collector.collect(self.agent, self.device)

            mean_reward = (
                torch.tensor(
                    [
                        (
                            episode[2].mean()
                            if self.reward_mode == "mean"
                            else episode[2].sum()
                        )
                        for episode in env_data
                    ]
                )
                .cpu()
                .mean()
            )

            self.tensorboard.add_scalar(
                f"metrics/reward_{mode}", mean_reward, self.agent.global_step
            )

            if mode == "train":
                self.agent(env_data)

            if delta_time > self.tqdm_reward_update_s and len(self.rewards_) > 0:
                start_time += self.tqdm_reward_update_s
                delta_time = 0
                pbar.set_description(f"Step: {step} | Reward: {mean_reward:<.4f}")

    def _reset_stats(self):
        self.rewards_ = []
        self.dones_ = []

    @classmethod
    def _resolve_path(cls, path: str | Path):
        if isinstance(path, str):
            path = Path(path).resolve()

        if path.is_file():
            path = path.with_suffix("")

        path.mkdir(exist_ok=True)

        return path

    def save(self, path: str | Path = AIPaths.cache):
        path = Trainer._resolve_path(path)

        name = type(self.agent).__name__

        params = {}
        params["env"] = self.env.spec.id
        params["learner"] = name

        with open(path / "params.json", "w+") as f:
            torch.save(self.agent.state_dict(), path / f"{name}.pt")
            f.write(json.dumps(params))

    @classmethod
    def load(cls, path: str | Path, render_mode: str = "human"):
        path = cls._resolve_path(path)

        params: dict = {}
        with open(path / "params.json") as f:
            params = json.loads(f.read())

        env_id = params["env"]
        env = gym.make(env_id, render_mode=render_mode)

        agent = torch.load(path / "agent.module")

        return Trainer(env, agent)

    def plot_rewards(self):
        sns.lineplot(self.rewards_)
        plt.title("Rewards per episode")
        plt.ylabel("Rewards")
        plt.xlabel("Number of episodes")
        plt.plot()


# class Trainer:
#     """
#     Collects data from the environment and trains the agent.
#     """

#     def __init__(
#         self,
#         collector: Collector,
#         agent: Learner,
#         tqdm_reward_update_s: float = 0.2,
#         reward_svg_size: int = 50,
#         reward_mode: Literal["sum", "mean"] = "mean",
#     ):
#         self.collector = collector
#         self.agent = agent
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.tqdm_reward_update_s = tqdm_reward_update_s
#         self.reward_svg_size = reward_svg_size
#         self.rewards_ = []
#         self.train_mode_ = None
#         self.run_name = random_run_name()
#         self.tensorboard = SummaryWriter(AIPaths.tensorboard / self.run_name)
#         self.agent.set_logger(self.tensorboard)
#         self.reward_mode = reward_mode

#     def train(self):
#         # Make sure we are in a train render mode
#         if self.train_mode_ is None:
#             self.train_mode_ = self.collector.render_mode
#         if self.train_mode_ != self.collector.render_mode:
#             self.collector.set_render(self.train_mode_)

#         self.agent.to(self.device)
#         self.agent.train()

#         return self._run("train")

#     def test(self, steps=10000):
#         self.agent.eval()
#         return self._run("test", steps)

#     def _run(self, mode="train", steps=None):
#         self._reset_stats()

#         state, _ = self.env.reset(seed=42)
#         state = torch.from_numpy(state).float()
#         local_steps = 0
#         rewards = MeanMetric() if self.reward_mode == "mean" else SumMetric()
#         pbar = tqdm(range(int(steps)))

#         start_time = time()
#         delta_time = 0

#         for step in pbar:
#             action = self.agent.act(state)

#             parsed_action = None
#             if isinstance(self.env.action_space, Discrete):
#                 parsed_action = action.cpu().argmax().numpy()
#             else:
#                 parsed_action = action.cpu().numpy()

#             next_state, reward, terminated, truncated, _ = self.env.step(parsed_action)
#             next_state = torch.from_numpy(next_state).float()

#             done = terminated or truncated

#             rewards.update(reward)

#             if mode == "train":
#                 self.agent.learn(state, action, reward, next_state, done)

#             state = next_state

#             if done:
#                 reward_metric = rewards.compute().item()
#                 self.tensorboard.add_scalar(
#                     f"metrics/reward_{mode}", reward_metric, self.agent.global_step
#                 )
#                 self.rewards_.append(reward_metric)
#                 rewards.reset()
#                 state, _ = self.env.reset()
#                 state = torch.from_numpy(state).float()

#             self.agent.step(step)
#             local_steps += 1
#             delta_time = time() - start_time

#             if delta_time > self.tqdm_reward_update_s and len(self.rewards_) > 0:
#                 start_time += self.tqdm_reward_update_s
#                 delta_time = 0
#                 mean = float(np.mean(self.rewards_[-self.reward_svg_size :]))

#                 pbar.set_description(f"Step: {step} | Reward: {mean:<.4f}")

#         self.env.close()
#         return self.plot_rewards()

#     def _reset_stats(self):
#         self.rewards_ = []
#         self.dones_ = []

#     @classmethod
#     def _resolve_path(cls, path: str | Path):
#         if isinstance(path, str):
#             path = Path(path).resolve()

#         if path.is_file():
#             path = path.with_suffix("")

#         path.mkdir(exist_ok=True)

#         return path

#     def save(self, path: str | Path = AIPaths.cache):
#         path = Trainer._resolve_path(path)

#         name = type(self.agent).__name__

#         params = {}
#         params["env"] = self.env.spec.id
#         params["learner"] = name

#         with open(path / "params.json", "w+") as f:
#             torch.save(self.agent.state_dict(), path / f"{name}.pt")
#             f.write(json.dumps(params))

#     @classmethod
#     def load(cls, path: str | Path, render_mode: str = "human"):
#         path = cls._resolve_path(path)

#         params: dict = {}
#         with open(path / "params.json") as f:
#             params = json.loads(f.read())

#         env_id = params["env"]
#         env = gym.make(env_id, render_mode=render_mode)

#         agent = torch.load(path / "agent.module")

#         return Trainer(env, agent)

#     def plot_rewards(self):
#         sns.lineplot(self.rewards_)
#         plt.title("Rewards per episode")
#         plt.ylabel("Rewards")
#         plt.xlabel("Number of episodes")
#         plt.plot()

#     @property
#     def mean_rewards_(self):
#         return np.mean(self.rewards_)
