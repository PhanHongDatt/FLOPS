"""MLflow rejects metric names with '(' ')' — Ultralytics emits 'metrics/precision(B)'.

Kaggle s2 v2 trained G2 for 2 h 10 min, then crashed in log_metrics on exactly
that name, before evaluation and before FedAvg/FedProx ran.
"""
from __future__ import annotations

import math
import sys
from types import SimpleNamespace

from src.utils import logger


def test_metric_names_are_sanitised_and_non_finite_dropped(monkeypatch):
    seen = {}
    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(
        log_metrics=lambda m, step=None: seen.update(m)))
    logger.log_metrics({"train_metrics/precision(B)": 0.5, "mAP50-95": 0.3,
                        "bad": float("nan"), "lr/pg0": 0.01})
    assert seen == {"train_metrics/precision_B_": 0.5, "mAP50-95": 0.3, "lr/pg0": 0.01}
    assert all(not math.isnan(v) for v in seen.values())


def test_mlflow_failure_does_not_abort_the_run(monkeypatch, caplog):
    """Bookkeeping must never cost a finished training run."""
    def boom(m, step=None):
        raise RuntimeError("tracking store unavailable")
    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(log_metrics=boom))
    logger.log_metrics({"mAP50": 0.1})   # must not raise
