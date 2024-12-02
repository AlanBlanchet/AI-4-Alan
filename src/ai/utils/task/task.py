from pathlib import Path

from ...configs.main import MainConfig
from ...task.task import Task
from ...utils.env import AIEnv


def run_task(
    config: Path,
    extra_params={},
    default_run=AIEnv.run_configs_p / "default.yml",
):
    # Resolve config
    config = AIEnv.resolve_config(
        config, extra_params=extra_params, default_run=default_run
    )

    # Create the main config object
    config = MainConfig(**config)

    Task.run(config)
