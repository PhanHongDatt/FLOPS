from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.utils.artifacts import make_run_dir, save_environment, write_run_readme, verify_artifacts
from src.utils.config import load_experiment_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _read_environment_lock() -> dict[str, Any]:
    lock_path = Path(__file__).parents[2] / "environment.lock"
    if not lock_path.exists():
        logger.warning("environment.lock not found — environment metadata incomplete")
        return {}
    with lock_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def prepare_run(
    exp_id: str,
    config_path: Path,
    run_class: str,   # smoke | feasibility | main
    seed: int,
    partition_id: str,
) -> tuple[Path, dict[str, Any]]:
    if run_class not in ("smoke", "feasibility", "main"):
        raise ValueError(f"Invalid run_class: {run_class}")

    config = load_experiment_config(config_path)
    run_id = f"{exp_id}_{run_class}_seed{seed}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
    run_dir = make_run_dir(run_id)

    env = {
        **_read_environment_lock(),
        "git_commit": _git_commit(),
        "seed": seed,
        "partition_id": partition_id,
        "run_class": run_class,
        "exp_id": exp_id,
    }
    save_environment(run_dir, env)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    write_run_readme(run_dir, {
        "exp_id": exp_id,
        "run_class": run_class,
        "seed": seed,
        "partition_id": partition_id,
        "git_commit": env.get("git_commit", "unknown"),
        "started": datetime.utcnow().isoformat(),
    })

    logger.info("Run prepared: %s → %s", run_id, run_dir)
    return run_dir, config


def finalize_run(run_dir: Path) -> None:
    missing = verify_artifacts(run_dir)
    if missing:
        logger.warning("Run %s is missing artifacts: %s", run_dir.name, missing)
    else:
        logger.info("Run %s artifact check passed.", run_dir.name)
