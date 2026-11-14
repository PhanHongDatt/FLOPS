"""ADR-036 degradation-localisation probe: pure helpers."""
from __future__ import annotations

import numpy as np

from src.evaluation.probe import (
    balanced_index,
    centre_cells,
    confusion,
    decide,
    fit_probe,
    image_half,
    keep_box,
    paired_image_bootstrap,
    primary_level,
    recall_metrics,
)


def test_keep_box_is_deterministic_and_samples_cars_only():
    imgs = [f"im{i}" for i in range(4000)]
    cars = [keep_box(im, 0, 0) for im in imgs]
    assert cars == [keep_box(im, 0, 0) for im in imgs]
    assert 0.06 < np.mean(cars) < 0.10
    assert all(keep_box(im, 3, 1) and keep_box(im, 3, 2) for im in imgs[:50])
    assert not keep_box("x", 0, 3)


def test_image_half_splits_by_image_roughly_evenly():
    halves = [image_half(f"im{i}") for i in range(4000)]
    assert 0.45 < np.mean(halves) < 0.55
    assert image_half("abc") == image_half("abc")


def test_primary_level_and_centre_cells():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 100, 100], [0, 0, 300, 300], [950, 530, 960, 544]], float)
    assert primary_level(boxes).tolist() == [0, 1, 2, 0]
    cells = centre_cells(boxes, [(68, 120), (34, 60), (17, 30)])
    assert cells[0, 0].tolist() == [0, 0]
    assert cells[1, 1].tolist() == [3, 3]                 # centre 50 / 16
    assert cells[3, 2].tolist() == [16, 29]               # clipped to the grid


def test_balanced_index():
    y = np.array([0] * 50 + [1] * 5 + [2] * 9)
    idx = balanced_index(y)
    assert np.bincount(y[idx]).tolist() == [5, 5, 5]
    assert idx.tolist() == balanced_index(y).tolist()


def test_probe_separates_separable_classes():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [4, 0], [0, 4]], float)
    y = np.repeat([0, 1, 2], 100)
    x = centres[y] + rng.normal(0, 0.5, (300, 2))
    p = fit_probe(x, y)
    pred = p.predict(x)
    assert (pred == y).mean() > 0.97
    m = recall_metrics(y, pred == y)
    assert m["bus_recall"] > 0.95 and confusion(y, pred).sum() == 300


def test_paired_bootstrap_detects_a_shift_and_not_noise():
    images = np.repeat(np.arange(400), 3).astype(str)
    mask = np.ones(1200, bool)
    a = np.zeros(1200)
    b = np.r_[np.ones(300), np.zeros(900)]
    rng = np.random.default_rng(1)
    b = b[rng.permutation(1200)]
    r = paired_image_bootstrap(images, a, b, mask, n=300)
    assert abs(r["delta"] - 0.25) < 1e-9 and r["ci_low"] > 0
    r0 = paired_image_bootstrap(images, a, a, mask, n=100)
    assert r0["delta"] == 0 and r0["ci_low"] <= 0 <= r0["ci_high"]


def test_decide_follows_the_rule_and_requires_replication():
    hit = {"delta": 0.05, "ci_low": 0.01, "ci_high": 0.09}
    miss = {"delta": 0.01, "ci_low": -0.02, "ci_high": 0.04}
    big_but_noisy = {"delta": 0.05, "ci_low": -0.01, "ci_high": 0.1}
    feat = {"neck": hit, "cv3_1": hit, "readout": hit}
    head = {"neck": miss, "cv3_1": hit, "readout": hit}
    read = {"neck": big_but_noisy, "cv3_1": miss, "readout": hit}
    none = {"neck": miss, "cv3_1": miss, "readout": miss}
    assert decide({"11m": feat, "11s": feat}) == "features degraded"
    assert decide({"11m": head, "11s": head}) == "head body degraded"
    assert decide({"11m": read, "11s": read}) == "readout shifted"
    assert decide({"11m": none, "11s": none}) == "not a classification-separability effect"
    assert decide({"11m": feat, "11s": none}).startswith("not replicated")
