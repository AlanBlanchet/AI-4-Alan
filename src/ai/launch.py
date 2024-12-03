from copy import deepcopy
from pathlib import Path

from .configs import Base
from .dataset import BaseDataset
from .task.task import Task
from .train.runner import Runner
from .utils.env import AIEnv


class Main(Base):
    task: Task
    dataset: BaseDataset
    run: Runner
    config: dict

    def __call__(self):
        # Save this root config in the instances so they can communicate
        self.task._root_config = self
        self.dataset._root_config = self
        self.run._root_config = self
        # Run the pipeline
        self.run()


def run_task(
    config: Path,
    extra_params={},
    default_run=AIEnv.run_configs_p / "default.yml",
):
    # Resolve config
    config = AIEnv.resolve_config(
        config, extra_params=extra_params, default_run=default_run
    )
    original_config = deepcopy(config)

    ds = BaseDataset.from_config(config["dataset"], deepcopy(original_config))
    config["dataset"] = ds
    config["task"]["dataset"] = ds

    task = Task.from_config(config["task"], deepcopy(original_config))
    config["task"] = task
    config["run"]["task"] = task

    config["run"] = Runner(**config["run"])
    config["config"] = original_config

    # Create the main config object
    config = Main(**config)
    config()
