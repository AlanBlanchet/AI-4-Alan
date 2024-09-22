from pathlib import Path

import yaml
from click import Path as CPath
from click import command, option
from questionary import autocomplete


@command(help="Run a task")
@option(
    "--config",
    "-c",
    help="Use a specific config",
    type=CPath(exists=True, path_type=Path),
    default=None,
)
def main(config: Path):
    from ..configs.main import MainConfig
    from ..task.task import Task
    from ..utils.env import AIEnv

    if config is None:
        config_ps = [AIEnv.from_root(p) for p in AIEnv.configs_p.rglob("*.yml")]

        config = autocomplete("Chose a config", choices=config_ps)

    config = yaml.full_load(config.read_text())

    config = MainConfig(**config)

    # AI.from_config(config).run()
    Task.run(config)


if __name__ == "__main__":
    main()
