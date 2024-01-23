from pathlib import Path


class AIPaths:
    root_p = Path(__file__).parents[3].resolve()

    cache = Path.home() / ".cache/ai"

    ai = root_p / "src/ai"
    rl = ai / "nn/rl"
    rl_envs = rl / "env/vendor"

    tensorboard = root_p / "runs"


AIPaths.cache.mkdir(exist_ok=True, parents=True)
