from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, field_validator

from .base import Base
from .models import Backbone


class ConstructConfig(BaseModel):
    type: str


class BackboneConfig(ConstructConfig):
    class Config:
        extra = "allow"

    def build(self, **kwargs) -> Backbone:
        return Base.from_config({**self.model_dump(), **kwargs})


class ModelConfig(ConstructConfig):
    class Config:
        extra = "allow"


class TaskConfig(BaseModel):
    class Config:
        extra = "allow"

    type: Literal["classification"] = "classification"
    input: str
    val_only: bool = False
    metric_val_only: bool = False
    model: ModelConfig


class ActionConfig(IntEnum):
    fit = 0
    val = 1


class DatasetSplitConfig(BaseModel):
    name: str = "train"
    size: float = 1.0


class DatasetConfig(BaseModel):
    source: Literal["hf"] = "hf"
    params: dict
    train: DatasetSplitConfig
    val: DatasetSplitConfig


class RunConfig(BaseModel):
    action: ActionConfig
    lightning: dict = {}
    datamodule: dict = {}

    @field_validator("action", mode="before")
    def _flexible_action(cls, v):
        match v:
            case "train":
                return ActionConfig.fit
            case "eval":
                return ActionConfig.val
            case _:
                return ActionConfig[v]


class MainConfig(BaseModel):
    task: TaskConfig
    dataset: DatasetConfig
    run: RunConfig
