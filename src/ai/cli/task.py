from pathlib import Path

from click import UNPROCESSED
from click import Path as CPath
from click_extra import argument, extra_command


@extra_command(
    params=None,
    context_settings=dict(
        ignore_unknown_options=True,
    ),
)
@argument("action", type=str, default="val")
@argument("config", type=CPath(exists=True, path_type=Path))
@argument("args", nargs=-1, type=UNPROCESSED)
def main(action: str, config: Path, args: list[str]):
    """Launch a task from a config file."""
    from ..launch import run_task
    from ..utils.env import AIEnv

    extra_params = AIEnv.parse_extra_args(*args, f"run.action={action}")
    run_task(config, extra_params=extra_params)


if __name__ == "__main__":
    main()
