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


def build_local_train_config(config: dict[str, Any]) -> dict[str, Any]:
    """Per-client local-training settings from a merged experiment config.

    Single source for FL clients (run_fl_experiment.py) and F3 (run_f3.py): F3's
    Δθ only says something about FL if its local training is the same update.
    """
    train = config["train"]
    return {
        "local_epochs": config["federated"]["local_epochs"],
        "batch_size": train["batch_size"],
        "image_size": config["model"]["image_size"],
        "lr0": train["lr0"],
        "conf": config["evaluation"]["conf"],
        "iou": config["evaluation"]["iou"],
        "device": train["device"],
        # Local-training knobs — see yolo_wrapper.train_one_round for why these
        # defaults differ from Ultralytics' own (plan.md §13 K2, §6.5 H1/H3).
        "workers": train.get("workers", 2),
        "warmup_epochs": train.get("warmup_epochs", 0.0),
        "close_mosaic": train.get("close_mosaic", 0),
        "client_run_val": train.get("client_run_val", False),
        "deterministic": train.get("deterministic", True),
        "nbs": train.get("nbs"),
    }
