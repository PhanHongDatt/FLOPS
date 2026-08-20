from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import mlflow


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    return logger


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str = "http://mlflow:5000",
    run_name: str | None = None,
) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    return run.info.run_id


def log_params(params: dict[str, Any]) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str | Path) -> None:
    mlflow.log_artifact(str(path))


def end_run() -> None:
    mlflow.end_run()
