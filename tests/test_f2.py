"""Tests for F2 controlled perturbation (CLAUDE.md §8 F2).

Perturbations must touch ONLY the intended parameters; the runner must restore
the unperturbed model between specs and record the effect matrix.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest
import yaml

from src.experiments.f2 import (
    F2Hooks,
    F2Spec,
    build_specs,
    class_row_bias_shift,
    class_row_noise,
    run_perturbations,
    shared_noise,
)

CLASSES = ("car", "bus", "truck", "motorcycle")
NC = len(CLASSES)
BUS = 1
SHARED = [f"model.22.cv3.{s}.1.conv.weight" for s in range(3)]
HEAD = [f"model.22.cv3.{s}.2.{k}" for s in range(3) for k in ("weight", "bias")]


def _state() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    state = {"model.0.conv.weight": rng.normal(size=(4, 3, 3, 3))}
    for s in range(3):
        state[f"model.22.cv3.{s}.1.conv.weight"] = rng.normal(size=(8, 8, 3, 3))
        state[f"model.22.cv3.{s}.1.bn.running_var"] = np.ones(8)
        state[f"model.22.cv3.{s}.2.weight"] = rng.normal(size=(NC, 8, 1, 1))
        state[f"model.22.cv3.{s}.2.bias"] = rng.normal(size=(NC,))
    return state


def _changed(a: dict, b: dict) -> set[str]:
    return {k for k in a if not np.array_equal(a[k], b[k])}


def test_row_noise_touches_only_target_weight_row():
    """Bias is excluded: a bias "row" is one scalar (~-8.5 after bias_init), so
    rel x RMS noise on it would shift the logit by several units and swamp the
    weight-row effect. Bias is probed separately by row_bias_shift."""
    base = _state()
    new, l2 = class_row_noise(base, NC, BUS, rel=1.0, seed=0)
    weights = [k for k in HEAD if k.endswith(".weight")]
    assert _changed(base, new) == set(weights)
    for k in weights:
        others = [c for c in range(NC) if c != BUS]
        np.testing.assert_array_equal(new[k][others], base[k][others])
    assert l2 > 0
    assert _changed(base, _state()) == set()  # input not mutated


def test_row_noise_is_seeded_and_scales_with_rel():
    base = _state()
    a, l2a = class_row_noise(base, NC, BUS, rel=1.0, seed=3)
    b, _ = class_row_noise(base, NC, BUS, rel=1.0, seed=3)
    _, l2_half = class_row_noise(base, NC, BUS, rel=0.5, seed=3)
    assert _changed(a, b) == set()
    assert l2_half == pytest.approx(l2a / 2)


def test_bias_shift_moves_only_target_bias():
    base = _state()
    new, l2 = class_row_bias_shift(base, NC, BUS, delta=2.0)
    assert _changed(base, new) == {k for k in HEAD if k.endswith("bias")}
    for s in range(3):
        k = f"model.22.cv3.{s}.2.bias"
        assert new[k][BUS] == pytest.approx(base[k][BUS] - 2.0)
    assert l2 == pytest.approx(2.0 * np.sqrt(3))


def test_shared_noise_is_norm_matched_and_avoids_head_and_bn():
    base = _state()
    new, l2 = shared_noise(base, target_l2=0.7, seed=1)
    assert _changed(base, new) == set(SHARED)
    assert l2 == pytest.approx(0.7)


def test_perturbations_fail_loudly_without_class_head():
    with pytest.raises(ValueError, match="exactly 6"):
        class_row_noise({"model.0.conv.weight": np.zeros((2, 2))}, NC, BUS, rel=1.0, seed=0)


def test_build_specs_grid():
    cfg = {"noise_rel": [0.5, 1.0], "bias_shift": [1.0], "seeds": [0, 1], "shared_control": True}
    specs = build_specs(cfg, CLASSES, target_class="bus")
    kinds = [s.kind for s in specs]
    assert kinds.count("row_noise") == NC * 2 * 2
    assert kinds.count("row_bias_shift") == NC * 1
    assert kinds.count("shared_noise") == 2 * 2  # per level x seed, matched to the target row
    assert len({s.spec_id for s in specs}) == len(specs)


class _FakeModel:
    def __init__(self, state):
        self.state = {k: np.array(v, copy=True) for k, v in state.items()}


def _fake_hooks(loads: list) -> F2Hooks:
    def evaluate(model) -> dict[str, float]:
        # AP of class c falls with the drift of its own bias row only
        base = _state()
        out = {}
        for c, name in enumerate(CLASSES):
            d = sum(abs(model.state[f"model.22.cv3.{s}.2.bias"][c] - base[f"model.22.cv3.{s}.2.bias"][c])
                    for s in range(3))
            out[f"AP50_{name}"] = 0.8 - 0.1 * d
        return out

    def load(model, state) -> None:
        loads.append(state)
        model.state = {k: np.array(v, copy=True) for k, v in state.items()}

    return F2Hooks(
        build=lambda: _FakeModel(_state()),
        state=lambda m: {k: np.array(v, copy=True) for k, v in m.state.items()},
        load=load,
        evaluate=evaluate,
        confidence=lambda m: {"mean_conf_bus": 0.5},
    )


def test_runner_effect_matrix_and_artifacts(tmp_path):
    cfg = {"noise_rel": [], "bias_shift": [1.0], "seeds": [0], "shared_control": False}
    specs = build_specs(cfg, CLASSES, target_class="bus")
    loads: list = []
    summary = run_perturbations(specs, CLASSES, _fake_hooks(loads), tmp_path)

    m = summary["effect_matrix"]["row_bias_shift"]["1.0"]
    assert m["bus"]["bus"]["mean"] == pytest.approx(-0.3)   # 3 scales x 1.0 x 0.1
    assert m["bus"]["car"]["mean"] == pytest.approx(0.0)
    # the model is reset to the base state after the last spec
    assert _changed(loads[-1], _state()) == set()

    rows = list(csv.DictReader((tmp_path / "metrics.csv").open()))
    assert {r["spec_id"] for r in rows} == {s.spec_id for s in specs}
    log = yaml.safe_load((tmp_path / "perturbation_log.yaml").read_text())
    assert all(entry["l2"] > 0 for entry in log)
    assert "baseline" in yaml.safe_load((tmp_path / "summary.yaml").read_text())


def test_runner_rejects_target_not_in_classes():
    with pytest.raises(ValueError, match="target_class"):
        build_specs({"noise_rel": [1.0], "seeds": [0]}, CLASSES, target_class="person")


def test_spec_id_is_readable():
    s = F2Spec(kind="row_noise", row="bus", level=0.5, seed=2)
    assert s.spec_id == "row_noise_bus_0.5_s2"


def test_invalid_specs_are_rejected(tmp_path):
    hooks = _fake_hooks([])
    for bad in (F2Spec("row_noise", "bus", 1.0, None), F2Spec("row_noise", None, 1.0, 0),
                F2Spec("dropout", "bus", 1.0, 0)):
        with pytest.raises(ValueError):
            run_perturbations([bad], CLASSES, hooks, tmp_path / bad.kind)
