from __future__ import annotations

import logging
import math
import re
import sys
from pathlib import Path
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    return logger


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str = "http://mlflow:5000",
    run_name: str | None = None,
) -> str:
    import mlflow  # lazy -- only needed when actually running experiments
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name)
    return run.info.run_id


def log_params(params: dict[str, Any]) -> None:
    import mlflow
    mlflow.log_params(params)


_MLFLOW_NAME_BAD = re.compile(r"[^A-Za-z0-9_\-. :/]")


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log to MLflow with names it accepts; never let tracking abort a run.

    Ultralytics emits names such as ``metrics/precision(B)``; MLflow allows only
    alphanumerics, ``_ - . : /`` and spaces. Kaggle s2 v2 lost a finished 2 h G2
    training run to that MlflowException. Non-finite values are dropped.
    Results of record are the CSV artifacts, not MLflow.
    """
    import mlflow
    clean = {
        _MLFLOW_NAME_BAD.sub("_", str(k)): float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and math.isfinite(float(v))
    }
    try:
        mlflow.log_metrics(clean, step=step)
    except Exception as exc:  # tracking is auxiliary (CSV artifacts are canonical)
        logging.getLogger(__name__).warning("MLflow log_metrics failed (%s); continuing", exc)


def log_artifact(path: str | Path) -> None:
    import mlflow
    mlflow.log_artifact(str(path))


def end_run() -> None:
    import mlflow
    mlflow.end_run()
