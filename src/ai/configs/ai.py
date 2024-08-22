from copy import deepcopy

from ..task.task import Task
from ..train.trainer import AITrainer
from .base import BaseConfig


class AI(BaseConfig):
    trainer: AITrainer

    @classmethod
    def from_config(cls, config: dict):
        built_config = deepcopy(config)

        # ai_type = built_config.pop("_")

        # if ai_type == "train":
        trainer_config = built_config["trainer"]

        # Build blocks
        dataset = BaseConfig.from_config(built_config["dataset"])

        # Add dataset to task
        task_config = built_config["task"]
        task_config["dataset"] = dataset
        logger_config = task_config.pop("logger", None)
        if logger_config:
            task_config["logger"] = BaseConfig.from_config(logger_config)

        # Build task and model
        task: Task = BaseConfig.from_config(task_config)
        # model_cls: Type = BaseConfig.from_config(built_config["model"])

        # Task setup
        # model = task.setup(model_cls)

        # For trainer
        trainer_config_params = trainer_config.copy()
        trainer_config["task"] = task
        # trainer_config["model"] = AIModule(task=task)
        trainer_config_params.pop("_")
        trainer_config["params"] = trainer_config_params

        train_obj = BaseConfig.from_config(trainer_config)
        return cls(trainer=train_obj)

        # raise NotImplementedError(f"AI type {ai_type} not implemented")

    def run(self):
        self.trainer.run()
