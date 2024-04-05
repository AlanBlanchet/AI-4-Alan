import json
import os
from pathlib import Path
from time import time

import cv2
import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

from ai.utils.paths import AIPaths

from .agent.agent import Agent
from .env.environment import Environment
from .metrics.action import ActionMetric
from .utils.func import random_run_name


class Trainer:
    def __init__(
        self,
        agent: Agent,
        run_name: str = random_run_name(),
        device: torch.device | int | str = None,
        eval_steps=200,
        **kwargs,
    ):
        self.agent = agent
        self.eval_env = self.agent._env.clone(100)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.run_name = run_name

        self.run_p = AIPaths.runs_p / self.run_name
        self.vid_p = self.run_p / "vids"
        self.vid_p.mkdir(parents=True, exist_ok=True)

        self.logger = SummaryWriter(self.run_p)

        self.eval_steps = eval_steps
        self._train_steps = 0
        self.evals_ = 0
        self._loaded = False

        env = self.agent._env
        self.action_metric = ActionMetric(env.out_action, env.action_names)

        for k, v in kwargs.items():
            setattr(self, k, v)

        torch.autograd.set_detect_anomaly(True)

        # Prepare the agent
        self._prepare()

    def _prepare(self):
        [_ for _ in tqdm(self.agent.prepare(), desc="Preparing agent")]

    def start(self, steps=1000, **_):
        self.agent.train()
        self.agent.to(self.device)

        start_time = time()

        pbar = tqdm(range(int(steps)))

        for _ in pbar:
            step = self._train_steps
            delta_time = time() - start_time

            metrics = self.agent.learn()

            for k, v in metrics.items():
                self.logger.add_scalar(f"train/{k}", v, step)
            self.logger.add_scalar("train/epsilon", float(self.agent.epsilon), step)
            optim = self.agent._optim
            if optim is not None:
                self.logger.add_scalar("train/lr", optim.param_groups[0]["lr"], step)

            if step > 0 and step % self.eval_steps == 0:
                if not self.agent._env.simulated:
                    # Sample a train trajectory for plotting
                    traj = next(self.agent.trajectories(1))
                    obs = traj["obs"].cpu().repeat(1, 3, 1, 1).unsqueeze(0)
                    self.logger.add_video("train/trajectory", obs, step, fps=30)

                self.eval()
                self.agent.train()

            if delta_time > 0.2:
                pbar.set_description_str(
                    f"{' | '.join([f'{k}={v:.2f}' for k, v in metrics.items()])}"
                )
                start_time = time()

            self._train_steps += 1
            self.agent.step(step)

    def eval(self):
        # Current train env
        train_env = self.agent._env

        # Move the agent into the evaluation zone
        self.agent.eval(self.eval_env)
        self.eval_env.memory.clear()

        pbar = tqdm(leave=False)

        done = False
        i = 0
        self.agent._env.reset()
        obs = self.agent.view
        video_p = self.vid_p / f"eval_{self.evals_}.avi"
        writer = cv2.VideoWriter(
            str(video_p),
            cv2.VideoWriter_fourcc(*"MPEG"),
            30,
            obs.shape[:-1][::-1],
        )
        writer.write(obs)
        rewards = 0
        self.action_metric.reset()
        max_same_eval = self.agent._env.config.get("max_same_eval", -1)
        same_obs = 0
        while not done:
            _, action, reward, _, done, *_ = self.agent.interact()
            next_obs = self.agent.view

            if obs is not None and np.array_equal(obs, next_obs):
                same_obs += 1
            else:
                obs = next_obs
                same_obs = 0

            if max_same_eval > 0 and same_obs > max_same_eval:
                print("Stuck in eval, breaking...")
                break

            self.action_metric.update(action.argmax().item())
            rewards += reward
            pbar.update()
            i += 1
            writer.write(obs)
            pbar.set_description_str(f"Eval n°{self.evals_} - {i} ")

        self.logger.add_scalar("eval/reward", np.mean(rewards), self.evals_)

        for k, v in self.action_metric.compute().items():
            self.logger.add_scalar(f"action/{k}", v, self.evals_)

        # Can't directly use the h264 codec for vscode
        writer.release()
        os.system(
            f"ffmpeg -hide_banner -i {str(video_p)} -c:v libx264 {video_p.with_suffix('.mp4')} 2>/dev/null"
        )
        os.remove(video_p)
        self.evals_ += 1

        # Put the agent back into its training environment
        self.agent.drop(train_env)

    @classmethod
    def _resolve_path(cls, path: str | Path):
        if isinstance(path, str):
            if "/" not in path:
                path = AIPaths.cache_p / path
            else:
                path = Path(path).resolve()

        if path.is_file():
            path = path.with_suffix("")

        path.mkdir(exist_ok=True, parents=True)

        return path

    def save(self, path: str | Path = AIPaths.cache_p):
        path = Trainer._resolve_path(path)

        name = type(self.agent).__name__

        params = {}
        params["env"] = self.agent._env.env_name
        params["agent"] = name
        params["agent_mod"] = self.agent.__module__
        params["trainer"] = {
            "device": self.device.type,
            "eval_steps": self.eval_steps,
            "evals_": self.evals_,
            "run_name": self.run_name,
            "_train_steps": self._train_steps,
        }

        # Model
        torch.save(self.agent.state_dict(), path / "agent.pt")

        # Config
        with open(path / "params.json", "w+") as f:
            f.write(json.dumps(params))
        return path

    @classmethod
    def load(cls, path: str | Path):
        import importlib

        path = cls._resolve_path(path)

        agent_state = torch.load(path / "agent.pt")

        with open(path / "params.json", "r") as f:
            params = json.loads(f.read())
            trainer_params = params["trainer"]

        env = Environment(params["env"])

        # Load agent
        agent_name = str(params["agent"])
        agent_mod = str(params["agent_mod"])
        agent_mod = importlib.import_module(agent_mod)
        agent_class = getattr(agent_mod, agent_name)
        agent: Agent = agent_class(env)
        agent.load_state_dict(agent_state)

        trainer = Trainer(agent, **trainer_params)
        trainer._loaded = True
        return trainer
