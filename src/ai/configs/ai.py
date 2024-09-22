from copy import deepcopy

from ..task.task import Task
from ..train.runner import Runner, TrainerConfig
from .base import Base
from .main import MainConfig


class AI(Base):
    task: Task

    @classmethod
    def from_config(cls, config: dict):
        config = deepcopy(config)
        config: MainConfig = MainConfig(**config)

        task = Task(config)

        # Build blocks
        dataset = Base.from_config(built_config["dataset"])

        # Add dataset to task
        task_config = built_config["task"]
        task_config["dataset"] = dataset
        logger_config = task_config.pop("logger", None)
        if logger_config:
            task_config["logger"] = Base.from_config(logger_config)

        # Build task and model
        task: Task = Task.from_config(task_config)
        # model_cls: Type = BaseConfig.from_config(built_config["model"])

        # Task setup
        # model = task.setup(model_cls)

        # For trainer
        trainer_config_params = trainer_config.copy()
        trainer_config["task"] = task
        # trainer_config["model"] = AIModule(task=task)
        trainer_config_params.pop("_")
        trainer_config["config"] = TrainerConfig(**trainer_config_params)

        runner = Runner(config.run)

        runner = Base.from_config(trainer_config)
        return cls(runner=runner)

        # raise NotImplementedError(f"AI type {ai_type} not implemented")

    def run(self):
        self.trainer.run()
