import os
from functools import cache
from pathlib import Path

from .cache import SmartCache


class AIEnv:
    root_p = Path(__file__).parents[3].resolve()

    cache_p = Path.home() / ".cache/ai"

    fn_cache = SmartCache(cache_p / "fn_cache")

    ai_p = root_p / "src/ai"
    archs_p = ai_p / "nn/arch"
    configs_p = root_p / "configs"

    DEFAULT_NUM_PROC = max(2, os.cpu_count() - 2)

    @classmethod
    @cache
    def path2module(cls, path: Path) -> str:
        return "ai." + str(path.relative_to(cls.ai_p).with_suffix("")).replace("/", ".")

    @classmethod
    def from_root(self, path: Path) -> Path:
        return path.relative_to(self.root_p) if path.is_absolute() else path

    rl_p = ai_p / "nn/rl"
    rl_envs_p = rl_p / "env/vendor"

    runs_p = root_p / "runs"


AIEnv.cache_p.mkdir(exist_ok=True, parents=True)
AIEnv.runs_p.mkdir(exist_ok=True, parents=True)
