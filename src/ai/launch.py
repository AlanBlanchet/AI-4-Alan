from copy import deepcopy
from pathlib import Path

from pydantic import Field

from .configs import Base
from .dataset import BaseDataset
from .task.task import Task
from .train.runner import Runner
from .utils.env import AIEnv
from .utils.pydantic_ import validator


class Main(Base):
    config: dict
    dataset: BaseDataset = Field(None, validate_default=True)
    task: Task = Field(None, validate_default=True)
    run: Runner = Field(None, validate_default=True)

    @validator("dataset")
    def validate_dataset(cls, _, values):
        config = deepcopy(values["config"])
        return BaseDataset.from_config(config["dataset"], config)

    @validator("task")
    def validate_task(cls, _, values):
        config = deepcopy(values["config"])
        config["task"]["dataset"] = values["dataset"]
        return Task.from_config(config["task"], config)

    @validator("run")
    def validate_run(cls, _, values):
        config = deepcopy(values["config"])
        config["run"]["task"] = values["task"]
        return Runner(**config["run"])

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

    main = Main(config=config)
    main()
