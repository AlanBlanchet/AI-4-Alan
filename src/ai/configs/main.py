from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, field_validator

from ..utils.env import AIEnv
from .base import ModuleConfig
from .log import Loggable


class Config(BaseModel, Loggable):
    log_name: ClassVar[str] = "config"
    color: ClassVar[str] = "pink"


class TaskType(Config):
    name: str

    @property
    def alias(self) -> str:
        return f"{self.name}_alias"


class TaskConfig(Config):
    class Config:
        extra = "allow"

    type: Literal["classification", "detection", "reinforcementlearning"]
    val_only_metrics: bool = False
    model: ModuleConfig

    @property
    def required_fields(self):
        if self.type == "classification":
            return ["labels"]
        elif self.type == "detection":
            return ["labels", "bbox"]
        else:
            raise ValueError(f"Unknown task type {self.type}")


class ActionEnum(IntEnum):
    fit = 0
    val = 1


class DatasetSplitConfig(Config):
    name: str = "train"
    size: float = 1.0


class DatasetConfig(Config):
    class Config:
        extra = "allow"

    source: Literal["hf", "gym"] = "hf"
    params: dict
    input: str | list[str] | dict[str, Any]
    train: DatasetSplitConfig = {}
    val: DatasetSplitConfig = {}

    @property
    def map_params(self):
        _map = {"input": self.input, "id": "id"}
        for k, v in self.model_extra.items():
            if isinstance(v, str):
                _map.update({k: v})
            elif isinstance(v, dict):
                # Can also be a dict with name (support for transforms or other linked to the type)
                if "name" not in v:
                    v["name"] = k  # Default
                _map.update({k: v})
        return _map


class DataModuleConfig(Config):
    num_workers: int = AIEnv.DEFAULT_NUM_PROC
    pin_memory: bool = True
    batch_size: int = 1

    @property
    def persistent_workers(self):
        return self.num_workers > 0

    class Config:
        extra = "allow"


class RunConfig(Config):
    action: ActionEnum
    lightning: dict = {}
    datamodule: DataModuleConfig = DataModuleConfig()
    checkpoint: Path = None

    @field_validator("action", mode="before")
    def _flexible_action(cls, v):
        match v:
            case "train":
                return ActionEnum.fit
            case "eval":
                return ActionEnum.val
            case _:
                return ActionEnum[v]

    @property
    def val_only(self):
        return self.action == ActionEnum.val


class MainConfig(Config):
    task: TaskConfig
    dataset: DatasetConfig
    run: RunConfig

    @field_validator("run", mode="before")
    def validate_run(cls, v, values):
        # The checkpoint
        checkpoint = values.data.get("checkpoint", None)
        task = values.data["task"][""]

        if isinstance(v, bool):
            if v:
                main_config = values.get("run")
                if main_config and isinstance(main_config, MainConfig):
                    if main_config.task.val_only_metrics:
                        raise ValueError(
                            "Checkpoint cannot be set when val_only_metrics is True"
                        )
            else:
                v = None

        # if isinstance(v, str):

        return v
