import json
from pathlib import Path
from time import time

import cv2
import torch
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

from ai.utils.paths import AIPaths

from .agent.agent import Agent
from .env.environment import Environment
from .utils.func import random_run_name


class Trainer:
    def __init__(
        self,
        agent: Agent,
        run_name: str = random_run_name(),
        device: torch.device | int | str = None,
        **kwargs,
    ):
        self.agent = agent
        self.eval_env = self.agent._env.clone()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.run_name = run_name

        self.run_p = AIPaths.runs_p / self.run_name
        self.vid_p = self.run_p / "vids"
        self.vid_p.mkdir(parents=True, exist_ok=True)

        self.logger = SummaryWriter(self.run_p)

        self.eval_steps = 10
        self.evals_ = 0
        self._loaded = False

        for k, v in kwargs.items():
            setattr(self, k, v)

        torch.autograd.set_detect_anomaly(True)

    def start(self, steps=1000):
        self.agent.train()
        self.agent.to(self.device)

        start_time = time()

        pbar = tqdm(range(int(steps)))

        for agent_step in pbar:
            delta_time = time() - start_time

            metrics = self.agent.learn()

            for k, v in metrics.items():
                self.logger.add_scalar(k, v, agent_step)

            if agent_step > 0 and agent_step % self.eval_steps == 0:
                self.eval()
                self.agent.train()

            if delta_time > 0.2:
                pbar.set_description_str(
                    f"{' | '.join([f'{k}={v:.2f}' for k, v in metrics.items()])}"
                )
                start_time = time()

    def eval(self):
        # Current train env
        train_env = self.agent._env

        # Move the agent into the evaluation zone
        self.agent.eval(self.eval_env)

        pbar = tqdm(leave=False)

        done = False
        i = 0
        dones = 0
        writer = cv2.VideoWriter(
            str(self.vid_p / f"eval_{self.evals_}.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            30,
            (160, 210),
        )
        self.agent._env.reset()
        while not done:
            obs, *_, done = self.agent.interact()
            pbar.update()
            if done:
                dones += 1
            i += 1
            writer.write(obs)
            pbar.set_description_str(f"Eval n°{self.evals_} - {i} ")

        self.evals_ += 1

        # Put the agent back into its training environment
        self.agent.drop(train_env)

    @classmethod
    def _resolve_path(cls, path: str | Path):
        if isinstance(path, str):
            if "/" not in path:
                path = AIPaths.cache / path
            else:
                path = Path(path).resolve()

        if path.is_file():
            path = path.with_suffix("")

        path.mkdir(exist_ok=True)

        return path

    def save(self, path: str | Path = AIPaths.cache):
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
        }

        # Model
        torch.save(self.agent.state_dict(), path / "agent.pt")

        # Config
        with open(path / "params.json", "w+") as f:
            f.write(json.dumps(params))

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
