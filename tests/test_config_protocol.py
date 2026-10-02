"""Evaluation and centralized-training protocol (ADR-009)."""
from __future__ import annotations

from pathlib import Path

from src.utils.config import (
    build_centralized_train_kwargs,
    build_local_train_config,
    load_experiment_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _cfg(name: str) -> dict:
    return load_experiment_config(ROOT / "configs" / "experiments" / f"{name}.yaml")


def test_ap_is_computed_at_the_ultralytics_val_threshold():
    """conf=0.001 traces the whole PR curve; Ultralytics then fills the confusion
    matrix (FP/FN) at 0.25 by itself (utils/metrics.py ConfusionMatrix.process_batch)."""
    ev = _cfg("feasibility")["evaluation"]
    assert ev["conf"] == 0.001
    assert ev["operating_conf"] == 0.25


def test_local_train_config_carries_both_thresholds():
    tc = build_local_train_config(_cfg("feasibility"))
    assert tc["conf"] == 0.001 and tc["operating_conf"] == 0.25


def test_centralized_kwargs_forward_seed_and_schedule():
    """train_centralized.py never passed --seed: every G2 'seed' trained with seed 0."""
    cfg = _cfg("feasibility")
    kw = build_centralized_train_kwargs(cfg, seed=123)
    assert kw["seed"] == 123
    assert kw["epochs"] == cfg["train"]["epochs"]
    assert kw["batch"] == cfg["train"]["batch_size"]
    # multi-epoch centralized run uses Ultralytics' own schedule defaults
    assert kw["warmup_epochs"] == 3.0
    assert kw["close_mosaic"] == 10
    assert kw["workers"] == cfg["train"]["workers"]


def test_fl_local_training_keeps_round_settings():
    """One local epoch per round: warm-up / close_mosaic stay off for FL clients."""
    tc = build_local_train_config(_cfg("feasibility"))
    assert tc["warmup_epochs"] == 0.0 and tc["close_mosaic"] == 0


def test_feasibility_batch_matches_base_config():
    assert _cfg("feasibility")["train"]["batch_size"] == 16


def test_smoke_keeps_small_batch_without_accumulation():
    tr = _cfg("smoke")["train"]
    assert tr["batch_size"] == tr["nbs"] == 4
