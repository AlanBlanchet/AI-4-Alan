from pathlib import Path

import yaml
from click import Path as CPath
from click import command, option
from questionary import autocomplete


@command(help="Train a model")
@option(
    "--config",
    "-c",
    help="Use a specific config",
    type=CPath(exists=True, path_type=Path),
    default=None,
)
def main(config: Path):
    from dotenv import load_dotenv

    from ..configs.ai import AI

    load_dotenv()

    if config is None:
        config_ps = [AI.from_root(p) for p in AI.configs_p.rglob("*.yml")]

        config = autocomplete("Chose a config", choices=config_ps)

    config = yaml.full_load(config.read_text())

    AI.from_config(config).run()


if __name__ == "__main__":
    main()
