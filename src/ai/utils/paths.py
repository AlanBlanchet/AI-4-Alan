from pathlib import Path


class AIPaths:
    root_p = Path(__file__).parents[3].resolve()

    ai = root_p / "src/ai"
    rl = ai / "nn/rl"
    rl_envs = rl / "env/vendor"
