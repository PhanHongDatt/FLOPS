"""Tests for the F3 matched-pair runner (CLAUDE.md §8 F3).

Training/evaluation are injected, so these run without Ultralytics or a GPU.
The fake trainer encodes the hypothesised mechanism (a client without the
target class pushes that class's logit bias down) only to exercise the
plumbing — it is not evidence for it.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.partitioner import PartitionManifest, save_partition_manifest
from src.experiments.f3 import F3Hooks, resolve_side, run_matched_pair, validate_pair

CLASSES = ("car", "bus", "truck", "motorcycle")
NC = len(CLASSES)


def _manifest(tmp: Path, name: str, bus_count: int) -> Path:
    m = PartitionManifest(
        partition_id=name, seed=42, scenario="S1", num_clients=1,
        client_assignments={"C0": ["a.jpg"]},
        class_counts={"C0": {"car": 10, "bus": bus_count, "truck": 3, "motorcycle": 1}},
        missing_classes={"C0": [] if bus_count else ["bus"]},
    )
    path = tmp / name / "manifest.yaml"
    save_partition_manifest(m, path)
    (tmp / name / "data_C0.yaml").write_text("nc: 4\n")
    return path


def _global_state() -> dict[str, np.ndarray]:
    state = {"model.0.conv.weight": np.zeros((2, 2))}
    for s in range(3):
        state[f"model.22.cv3.{s}.2.weight"] = np.zeros((NC, 2, 1, 1))
        state[f"model.22.cv3.{s}.2.bias"] = np.full((NC,), 1.0)
    return state


def _fake_hooks(calls: list) -> F3Hooks:
    def build():
        return {"state": _global_state()}

    def train(model, data_yaml: Path, seed: int, project: Path, name: str) -> None:
        calls.append((data_yaml.parent.name, seed, name))
        shift = -0.5 if "s1b" in str(data_yaml) else 0.1
        for s in range(3):
            model["state"][f"model.22.cv3.{s}.2.bias"][1] += shift + 0.0001 * seed
        model["state"]["model.0.conv.weight"] += 0.1

    def evaluate(model) -> dict[str, float]:
        bus_bias = float(model["state"]["model.22.cv3.0.2.bias"][1])
        return {"mAP50": 0.5, "AP50_bus": max(0.0, 0.3 * bus_bias), "AP50_car": 0.6, "FP_bus": 5.0}

    def state(model) -> dict[str, np.ndarray]:
        return {k: np.array(v, copy=True) for k, v in model["state"].items()}

    return F3Hooks(build=build, train=train, evaluate=evaluate, state=state)


@pytest.fixture
def pair(tmp_path):
    missing = resolve_side("missing", _manifest(tmp_path, "s1b", 0), "C0", "bus")
    control = resolve_side("control", _manifest(tmp_path, "ctrl", 40), "C0", "bus")
    return missing, control


def test_resolve_side_reads_target_boxes_and_data_yaml(pair):
    missing, control = pair
    assert missing.target_boxes == 0 and control.target_boxes == 40
    assert missing.data_yaml.name == "data_C0.yaml"
    assert missing.partition_id == "s1b"


def test_resolve_side_unknown_client(tmp_path):
    with pytest.raises(KeyError, match="C9"):
        resolve_side("missing", _manifest(tmp_path, "s1b", 0), "C9", "bus")


def test_validate_pair_rejects_wrong_roles(tmp_path):
    has_bus = resolve_side("missing", _manifest(tmp_path, "a", 5), "C0", "bus")
    no_bus = resolve_side("control", _manifest(tmp_path, "b", 0), "C0", "bus")
    with pytest.raises(ValueError, match="zero"):
        validate_pair(has_bus, has_bus)
    with pytest.raises(ValueError, match="positive"):
        validate_pair(no_bus, no_bus)


def test_run_writes_artifacts_and_contrast(pair, tmp_path):
    missing, control = pair
    calls: list = []
    out = tmp_path / "out"
    summary = run_matched_pair(missing, control, "bus", CLASSES, seeds=[42, 123], hooks=_fake_hooks(calls), out_dir=out)

    # every seed trains both sides, from the same global model
    assert sorted((c[0], c[1]) for c in calls) == [("ctrl", 42), ("ctrl", 123), ("s1b", 42), ("s1b", 123)]
    for seed in (42, 123):
        assert (out / f"update_analysis_seed{seed}.yaml").exists()
    rows = list(csv.DictReader((out / "metrics.csv").open()))
    assert {r["side"] for r in rows} == {"missing", "control"}
    bus_rows = [r for r in rows if r["metric"] == "AP50_bus" and r["side"] == "missing"]
    assert all(float(r["delta"]) < 0 for r in bus_rows)

    gap = summary["parameter"]["bias_delta_gap"]
    assert gap["n"] == 2 and gap["mean"] == pytest.approx(-0.6)
    assert summary["prediction"]["AP50_bus"]["delta_gap"]["mean"] < 0
    assert summary["target_known_before"] is True
    assert yaml.safe_load((out / "summary.yaml").read_text())["target_class"] == "bus"


def test_run_flags_target_unknown_before(pair, tmp_path):
    """A head that never learned bus (e.g. raw COCO init: cv3 is re-initialised
    at nc=4) cannot show forgetting; the run must say so, not report a null effect."""
    missing, control = pair
    hooks = _fake_hooks([])
    blind = F3Hooks(build=hooks.build, train=hooks.train, state=hooks.state,
                    evaluate=lambda m: {"mAP50": 0.0, "AP50_bus": 0.0})
    summary = run_matched_pair(missing, control, "bus", CLASSES, seeds=[1], hooks=blind, out_dir=tmp_path / "o")
    assert summary["target_known_before"] is False


def test_run_rejects_nondeterministic_build(pair, tmp_path):
    missing, control = pair
    hooks = _fake_hooks([])
    rng = np.random.default_rng(0)

    def noisy_build():
        m = hooks.build()
        m["state"]["model.0.conv.weight"] += rng.normal(size=(2, 2))
        return m

    bad = F3Hooks(build=noisy_build, train=hooks.train, evaluate=hooks.evaluate, state=hooks.state)
    with pytest.raises(RuntimeError, match="same global"):
        run_matched_pair(missing, control, "bus", CLASSES, seeds=[1], hooks=bad, out_dir=tmp_path / "o")


def test_run_rejects_disjoint_metrics(pair, tmp_path):
    missing, control = pair
    hooks = _fake_hooks([])
    calls = iter(range(100))
    disjoint = F3Hooks(build=hooks.build, train=hooks.train, state=hooks.state,
                       evaluate=lambda m: {f"m{next(calls)}": 1.0})
    with pytest.raises(ValueError, match="no metric"):
        run_matched_pair(missing, control, "bus", CLASSES, seeds=[1], hooks=disjoint, out_dir=tmp_path / "o")
