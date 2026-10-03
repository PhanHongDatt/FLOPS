"""Server eval cadence: 40-round runs evaluate every k rounds (+ final), checkpoint every round."""
from __future__ import annotations

import numpy as np

import src.model.yolo_wrapper as yw
from src.federated.server import _build_centralized_evaluate_fn


def _fn(tmp_path, monkeypatch, eval_every, num_rounds=10):
    calls = []
    monkeypatch.setattr(yw, "build_model", lambda weights: object())
    monkeypatch.setattr(yw, "set_parameters", lambda m, p: None)
    monkeypatch.setattr(yw, "evaluate", lambda **kw: calls.append(1) or {"mAP50": 0.1})
    fn = _build_centralized_evaluate_fn(
        global_data_yaml=tmp_path / "data.yaml",
        eval_config={"image_size": 64, "conf": 0.001, "iou": 0.7, "device": "cpu",
                     "eval_every": eval_every},
        run_dir=tmp_path, num_rounds=num_rounds, prune_client_weights=False)
    return fn, calls


def test_skips_off_cadence_rounds_but_checkpoints(tmp_path, monkeypatch):
    fn, calls = _fn(tmp_path, monkeypatch, eval_every=5)
    params = [np.zeros(2)]
    assert fn(2, params, {}) is None                       # skipped
    assert (tmp_path / "checkpoint" / "global_round_002.npz").exists()
    assert fn(5, params, {}) is not None                   # on cadence
    assert fn(10, params, {}) is not None                  # final round always
    assert len(calls) == 2


def test_round_zero_and_default_cadence(tmp_path, monkeypatch):
    fn, calls = _fn(tmp_path, monkeypatch, eval_every=1)
    params = [np.zeros(2)]
    assert fn(0, params, {}) is not None and fn(3, params, {}) is not None
    assert len(calls) == 2
