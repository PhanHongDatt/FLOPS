"""Tests for parameter drift analysis (G4/F3 support)."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.drift import compute_delta, group_drift, summarize_drift


def _sample_state() -> dict[str, np.ndarray]:
    return {
        "model.0.conv.weight": np.ones((4, 3, 3, 3), dtype=np.float32),
        "model.5.cv1.conv.weight": np.ones((8, 4, 3, 3), dtype=np.float32),
        "model.15.cv1.conv.weight": np.ones((16, 8, 3, 3), dtype=np.float32),  # neck
        "model.22.cv2.0.conv.weight": np.ones((64, 16, 3, 3), dtype=np.float32),
        "model.22.cv3.0.conv.weight": np.ones((4, 16, 3, 3), dtype=np.float32),
        "model.22.dfl.conv.weight": np.ones((1, 16, 1, 1), dtype=np.float32),
    }


def test_zero_delta_when_states_equal():
    state = _sample_state()
    deltas = compute_delta(state, state)
    drifts = group_drift(deltas)
    for d in drifts:
        assert d.l2_norm == pytest.approx(0.0)


def test_delta_key_mismatch_raises():
    a = _sample_state()
    b = dict(a)
    b["model.99.extra"] = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError):
        compute_delta(b, a)


def test_delta_computed_per_group():
    global_state = _sample_state()
    local_state = {k: np.array(v, copy=True) for k, v in global_state.items()}
    # Only shift detect_cls; leave all other groups equal (delta = 0).
    local_state["model.22.cv3.0.conv.weight"] += 1.0

    deltas = compute_delta(local_state, global_state)
    drifts = {d.module_group: d for d in group_drift(deltas)}

    assert drifts["backbone"].l2_norm == 0
    assert drifts["neck"].l2_norm == 0
    assert drifts["detect_reg"].l2_norm == 0
    assert drifts["detect_cls"].l2_norm > 0


def test_group_drift_covers_all_present_groups():
    state = _sample_state()
    deltas = compute_delta({k: v + 0.1 for k, v in state.items()}, state)
    drifts = group_drift(deltas)
    groups = {d.module_group for d in drifts}
    assert groups == {"backbone", "neck", "detect_cls", "detect_reg", "detect_dfl"}


def test_summarize_drift_shape():
    state = _sample_state()
    deltas = compute_delta({k: v + 0.5 for k, v in state.items()}, state)
    summary = summarize_drift(group_drift(deltas))
    for g, metrics in summary.items():
        assert set(metrics.keys()) == {"l2_norm", "cosine_similarity", "num_params", "total_numel"}
