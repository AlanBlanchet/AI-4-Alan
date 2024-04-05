from __future__ import annotations

from .configs import ENV_CONFIGS


def get_preprocess(env_name: str):
    config = get_env_config(env_name)
    preprocess = config.get("preprocess")
    return preprocess if preprocess else lambda x: x


def get_env_config(env_name: str):
    env_name = env_name.lower()
    for k, v in ENV_CONFIGS.items():
        if k.lower() in env_name:
            return v
    return {}
