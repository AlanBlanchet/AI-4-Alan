from typing import Literal

from pydantic import BaseModel, field_validator

from ....dataset.env.environment import EnvironmentDataset
from ....utils.hyperparam import HYPERPARAM, Hyperparam


class DQNConfig(BaseModel):
    env: EnvironmentDataset
    optimizer: Literal["Adam", "RMSprop", "AdamW"] = "RMSProp"
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon: HYPERPARAM = Hyperparam(start=0.9, end=0.1)
    batch_size: int = 32
    history: int = 4
    reward_shaping: bool = True
    interactions_per_learn: int = 1
    # Double DQN
    target: int = 0
    tau: float = 0.995
    # DRQN
    recurrent: bool = False
    # Dual DQN
    duel: bool = False
    # PER
    per: bool = False

    @field_validator("history", mode="before")
    def validate_history(cls, value, values):
        history = max(value, 1)
        if len(values["env"].preprocessed_shape) <= 1:
            history = 1
        return history

    @property
    def requires_merge(self):
        return not self.recurrent
