from typing import Literal, get_args

DEFAULT_GYM_ENV_KEYS_TYPE = Literal["obs", "reward", "done", "info"]

DEFAULT_ENV_KEYS_TYPE = Literal["obs", "action", "reward", "next_obs", "done"]
DEFAULT_ENV_KEYS = get_args(DEFAULT_ENV_KEYS_TYPE)

DEFAULT_KEYS_TYPE = Literal[DEFAULT_ENV_KEYS_TYPE, "idx"]
DEFAULT_KEYS = get_args(DEFAULT_KEYS_TYPE)
