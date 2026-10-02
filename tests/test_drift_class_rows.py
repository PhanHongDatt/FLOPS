"""Tests for class-row drift and the F3 matched-pair update analysis.

CLAUDE.md §8 F3 asks for Δθ on the *validated class-associated group*, not only
per module. At nc=4 that group is 780 of the 370,962 elements in the cv3
branch, so a branch-level norm cannot show it.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.drift import (
    class_row_drift,
    compute_delta,
    group_drift,
    matched_pair_update_analysis,
)

CLASSES = ("car", "bus", "truck", "motorcycle")
NC = len(CLASSES)
BUS = CLASSES.index("bus")


def _state() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    state = {
        "model.0.conv.weight": rng.normal(size=(4, 3, 3, 3)),
        "model.15.cv1.conv.weight": rng.normal(size=(8, 4, 3, 3)),
        "model.22.cv2.0.2.weight": rng.normal(size=(64, 8, 1, 1)),
        "model.22.cv3.0.0.conv.weight": rng.normal(size=(8, 8, 3, 3)),
        "model.22.dfl.conv.weight": rng.normal(size=(1, 16, 1, 1)),
    }
    for s in range(3):
        state[f"model.22.cv3.{s}.2.weight"] = rng.normal(size=(NC, 8, 1, 1))
        state[f"model.22.cv3.{s}.2.bias"] = rng.normal(size=(NC,))
    return state


def _shift_row(state: dict[str, np.ndarray], cls: int, w: float, b: float) -> dict[str, np.ndarray]:
    out = {k: np.array(v, copy=True) for k, v in state.items()}
    for s in range(3):
        out[f"model.22.cv3.{s}.2.weight"][cls] += w
        out[f"model.22.cv3.{s}.2.bias"][cls] += b
    return out


def test_class_head_group_reported_alongside_modules():
    g = _state()
    local = _shift_row(g, BUS, 0.0, -1.0)
    drifts = {d.module_group: d for d in group_drift(compute_delta(local, g))}
    assert drifts["class_head"].total_numel == 3 * (NC * 8 + NC)
    assert drifts["class_head"].l2_norm == pytest.approx(np.sqrt(3.0))
    # existing module groups are unchanged in meaning: the head is part of detect_cls
    assert drifts["detect_cls"].l2_norm == pytest.approx(np.sqrt(3.0))
    assert drifts["backbone"].l2_norm == 0


def test_class_row_drift_isolates_rows():
    g = _state()
    rows = {r.class_name: r for r in class_row_drift(compute_delta(_shift_row(g, BUS, 0.5, -2.0), g), CLASSES)}
    assert rows["bus"].l2_norm > 0
    for c in ("car", "truck", "motorcycle"):
        assert rows[c].l2_norm == 0
    assert rows["bus"].bias_delta_mean == pytest.approx(-2.0)
    assert rows["bus"].weight_delta_mean == pytest.approx(0.5)
    # 3 scales x (8 weights + 1 bias)
    assert rows["bus"].numel == 27


def test_class_row_cosine_against_reference():
    g = _state()
    a = compute_delta(_shift_row(g, BUS, 1.0, 1.0), g)
    b = compute_delta(_shift_row(g, BUS, -1.0, -1.0), g)
    rows = {r.class_name: r for r in class_row_drift(a, CLASSES, reference=b)}
    assert rows["bus"].cosine_vs_reference == pytest.approx(-1.0)
    assert np.isnan(rows["car"].cosine_vs_reference)  # zero vector → undefined, not 0


def test_class_row_drift_fails_loudly_without_head():
    g = {"model.0.conv.weight": np.zeros((2, 2))}
    with pytest.raises(ValueError, match="exactly 6"):
        class_row_drift(compute_delta(g, g), CLASSES)


def test_matched_pair_contrast():
    """Missing client pushes bus logits down; control does not."""
    g = _state()
    missing = _shift_row(_shift_row(g, BUS, 0.0, -1.5), 0, 0.0, 0.1)
    control = _shift_row(_shift_row(g, BUS, 0.0, 0.2), 0, 0.0, 0.1)
    out = matched_pair_update_analysis(g, missing, control, target_class="bus", class_names=CLASSES)

    t = out["target_contrast"]
    assert t["target_class"] == "bus"
    assert t["bias_delta_mean_missing"] == pytest.approx(-1.5)
    assert t["bias_delta_mean_control"] == pytest.approx(0.2)
    assert t["bias_delta_gap"] == pytest.approx(-1.7)
    assert t["target_row_l2_ratio_missing"] > t["target_row_l2_ratio_control"]

    assert set(out["module_drift"]) == {"missing", "control", "cosine_missing_vs_control"}
    assert set(out["class_rows"]["missing"]) == set(CLASSES)
    assert out["module_drift"]["missing"]["backbone"]["l2_norm"] == 0


def test_matched_pair_rejects_unknown_target():
    g = _state()
    with pytest.raises(ValueError, match="target_class"):
        matched_pair_update_analysis(g, g, g, target_class="person", class_names=CLASSES)


def test_matched_pair_output_is_yaml_safe():
    import yaml

    g = _state()
    out = matched_pair_update_analysis(g, _shift_row(g, BUS, 0, -1), g, "bus", CLASSES)
    yaml.safe_dump(out)  # plain floats/str/dict only


def test_bn_running_stats_reported_separately():
    """BN running_mean/var are data statistics, not learned Δθ. Mixed into the
    module groups they dominated the detect_cls norm in the F3 smoke run."""
    g = _state()
    g["model.22.cv3.0.0.bn.running_var"] = np.ones(8)
    g["model.22.cv3.0.0.bn.num_batches_tracked"] = np.array(0)
    local = {k: np.array(v, copy=True) for k, v in g.items()}
    local["model.22.cv3.0.0.bn.running_var"] += 10.0
    local["model.22.cv3.0.0.bn.num_batches_tracked"] = np.array(7)
    drifts = {d.module_group: d for d in group_drift(compute_delta(local, g))}
    assert drifts["detect_cls"].l2_norm == 0
    assert drifts["bn_stats"].l2_norm == pytest.approx(np.sqrt(8 * 100.0))  # counter excluded
    assert drifts["bn_stats"].num_params == 1
