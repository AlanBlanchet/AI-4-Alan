from click import Path

from .env import AIEnv


def get_configs():
    """Get the configuration file path."""
    config_files = AIEnv.configs_p.rglob(".[yaml yml]")
    return list(config_files)


def resolve_current_config():
    """Resolve the current configuration file."""
    run_p = AIEnv.cache_p / "current_run"
    config_files = get_configs()
    if len(config_files) == 0:
        raise FileNotFoundError("No configuration files found.")
    return config_files[0]


def parse_config(config: Path):
    """Parse the configuration file and return the configuration dictionary."""
    config = {}
    with open(CONFIG_FILE, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            config[key] = value
    return config
