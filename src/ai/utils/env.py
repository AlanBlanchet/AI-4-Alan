import logging
import os
import tempfile
import warnings
from functools import cache
from pathlib import Path

import yaml
from deepmerge import always_merger

from .cache import SmartCache
from .dict import dict_from_dot_keys


class AIEnv:
    root_p = Path(__file__).parents[3].resolve()

    cache_p = Path.home() / ".cache/ai"

    fn_cache = SmartCache(cache_p / "fn_cache")

    ai_p = root_p / "src/ai"
    archs_p = ai_p / "nn/arch"
    configs_p = root_p / "configs"

    DEFAULT_NUM_PROC = max(2, os.cpu_count() - 2)

    run_configs_p = configs_p / "run"

    tmp_log_p = Path(tempfile.gettempdir()) / "tmp_log"

    @classmethod
    @cache
    def path2module(cls, path: Path) -> str:
        return "ai." + str(path.relative_to(cls.ai_p).with_suffix("")).replace("/", ".")

    @classmethod
    def from_root(self, path: Path) -> Path:
        return path.relative_to(self.root_p) if path.is_absolute() else path

    @classmethod
    def load(cls, path: str | Path) -> dict:
        path = Path(path)
        if path.suffix in [".yml", ".yaml"]:
            return yaml.full_load(path.read_text())
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def _recurrent_resolve_config(cls, config: Path):
        c = cls.load(config)
        merge = c.pop("merge", None)
        if merge is not None:
            if not isinstance(merge, list):
                merge = [merge]
            for m in merge:
                # Resolve absolute path or relative path
                from_config_p = AIEnv.configs_p / m
                sub_p = from_config_p
                if not (sub_p.exists() and sub_p.is_file()):
                    sub_p = config.parent / m
                # Merge configs
                c = always_merger.merge(cls._recurrent_resolve_config(sub_p), c)
        return c

    @classmethod
    def resolve_config(
        cls, config: Path, extra_params={}, default_run=run_configs_p / "default.yml"
    ):
        c = cls._recurrent_resolve_config(config)
        if "run" not in c:
            # Add the run config
            always_merger.merge(c, cls._recurrent_resolve_config(default_run))
        # Convert manual parameters normalized dict values
        formatted_params = dict_from_dot_keys(
            extra_params.keys(), extra_params.values()
        )
        always_merger.merge(c, formatted_params)
        return c

    @classmethod
    def parse_extra_args(cls, *args) -> dict:
        res = {}

        grouped_args = []
        param = False
        for arg in args:
            if param:
                grouped_args[-1] += f"={arg}"
                param = False
            else:
                if arg.startswith("--") and "=" not in arg:
                    param = True
                grouped_args.append(arg)

        for arg in grouped_args:
            unpacked = arg.split("=")
            if len(unpacked) < 2:
                raise ValueError(
                    f"Manually specified arguments must be in the form 'key=value'. Got '{arg}'"
                )
            k, *v = unpacked
            k = k.strip("-")
            v = "=".join(v)
            # try to convert to int or float
            try:
                v = float(v)
                if v.is_integer():
                    v = int(v)
            except Exception as _:
                ...

            res[k] = v
        return res

    rl_p = ai_p / "nn/rl"
    rl_envs_p = rl_p / "env/vendor"

    runs_p = root_p / "runs"


AIEnv.cache_p.mkdir(exist_ok=True, parents=True)
AIEnv.runs_p.mkdir(exist_ok=True, parents=True)
AIEnv.tmp_log_p.mkdir(exist_ok=True, parents=True)

warn_file_p = AIEnv.tmp_log_p / "user_future_warns.txt"


# Step 1: Set up a dedicated logger for specific warnings
warnings_logger = logging.getLogger("warnings_logger")
warnings_logger.setLevel(logging.WARNING)

# Create a file handler to log warnings to the specified file
file_handler = logging.FileHandler(warn_file_p)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
warnings_logger.addHandler(file_handler)

# Step 2: Save the original showwarning function
original_showwarning = warnings.showwarning


# Step 3: Define a custom showwarning function
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    if issubclass(category, (UserWarning, FutureWarning)):
        warnings_logger.warning(f"{category.__name__}: {message} ({filename}:{lineno})")
    else:
        original_showwarning(message, category, filename, lineno, file, line)


# Step 4: Override the warnings.showwarning with the custom function
warnings.showwarning = custom_showwarning
