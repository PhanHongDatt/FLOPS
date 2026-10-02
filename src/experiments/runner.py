from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.utils.artifacts import make_run_dir, save_environment, write_run_readme, verify_artifacts
from src.utils.config import load_experiment_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Track per-run FileHandler so finalize_run can detach cleanly and avoid
# duplicating log lines when multiple runs happen in the same Python process.
_RUN_LOG_HANDLERS: dict[str, logging.FileHandler] = {}


def _attach_run_log_handler(run_dir: Path) -> logging.FileHandler:
    """Route root logger records into <run_dir>/run.log per CLAUDE.md §21."""
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    _RUN_LOG_HANDLERS[str(run_dir.resolve())] = handler
    return handler


def _detach_run_log_handler(run_dir: Path) -> None:
    key = str(run_dir.resolve())
    handler = _RUN_LOG_HANDLERS.pop(key, None)
    if handler is None:
        return
    handler.flush()
    handler.close()
    logging.getLogger().removeHandler(handler)


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


def default_run_id(exp_id: str, run_class: str, seed: int, partition_id: str) -> str:
    """Deterministic run id — no timestamp.

    A timestamped id makes a run directory impossible to find again, which makes
    resume impossible (plan.md §13 K4). Determinism is what lets a Kaggle session
    that died at round 6 continue at round 7. Accidental overwrite is prevented
    downstream: ``run_fl_server`` refuses to start when checkpoints already exist
    unless ``resume=True``.
    """
    safe_partition = str(partition_id).replace("/", "_").replace("\\", "_")
    return f"{exp_id}_{run_class}_seed{seed}_{safe_partition}"


def prepare_run(
    exp_id: str,
    config_path: Path,
    run_class: str,   # smoke | feasibility | main
    seed: int,
    partition_id: str,
    run_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    if run_class not in ("smoke", "feasibility", "main"):
        raise ValueError(f"Invalid run_class: {run_class}")

    config = load_experiment_config(config_path)
    if run_id is None:
        run_id = default_run_id(exp_id, run_class, seed, partition_id)
    run_dir = make_run_dir(run_id)

    _attach_run_log_handler(run_dir)

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


def finalize_run(
    run_dir: Path,
    algorithm: str | None = None,
    run_type: str = "federated",
) -> None:
    missing = verify_artifacts(run_dir, algorithm=algorithm, run_type=run_type)
    if missing:
        logger.warning("Run %s is missing artifacts: %s", run_dir.name, missing)
    else:
        logger.info("Run %s artifact check passed.", run_dir.name)
    # Detach the per-run FileHandler last so verify_artifacts + summary logs
    # land in run.log before it's closed.
    _detach_run_log_handler(run_dir)
