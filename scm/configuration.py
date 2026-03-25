from copy import deepcopy
from pathlib import Path
import tomllib


CONFIG_DIR = Path(__file__).with_name("configs")
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.toml"


def deep_merge(base, override):
    """recursively merge override into base and return the merged dict."""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_run_config(path=None):
    """load the default SCM config, optionally merged with a user config."""

    with DEFAULT_CONFIG_PATH.open("rb") as f:
        config = tomllib.load(f)

    if path is None:
        return config

    user_path = Path(path)
    with user_path.open("rb") as f:
        override = tomllib.load(f)

    config = deep_merge(config, override)
    config["_config_path"] = str(user_path)
    return config

