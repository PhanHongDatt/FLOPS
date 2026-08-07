from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_experiment_config(exp_config_path: str | Path) -> dict[str, Any]:
    base_path = Path(__file__).parents[2] / "configs" / "base_config.yaml"
    base = load_config(base_path)
    override = load_config(exp_config_path)
    return merge_configs(base, override)


def resolve_env(config: dict[str, Any]) -> dict[str, Any]:
    """Replace ${ENV_VAR} placeholders in string config values."""
    result = {}
    for k, v in config.items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_key = v[2:-1]
            result[k] = os.environ.get(env_key, v)
        elif isinstance(v, dict):
            result[k] = resolve_env(v)
        else:
            result[k] = v
    return result
