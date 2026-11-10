"""AP50 per COCO size range (src/evaluation/size_split.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.size_split import ap50_in_range, size_split

SMALL = np.array([10.0, 10.0, 30.0, 30.0])      # 20 x 20 = 400 px²  (< 32²)
LARGE = np.array([100.0, 100.0, 300.0, 300.0])  # 200 x 200          (>= 96²)


def test_perfect_detections_give_ap_one_everywhere():
    gts = {"a": np.stack([SMALL, LARGE])}
    dets = [("a", 0.9, SMALL), ("a", 0.8, LARGE)]
    out = size_split(dets, gts)
    assert out["all"].ap50 == pytest.approx(1.0) and out["all"].n_gt == 2
    assert out["small"].ap50 == pytest.approx(1.0) and out["small"].n_gt == 1
    assert out["large"].ap50 == pytest.approx(1.0) and out["large"].n_gt == 1
    assert np.isnan(out["medium"].ap50) and out["medium"].n_gt == 0


def test_missed_small_object_only_hurts_the_small_range():
    gts = {"a": np.stack([SMALL, LARGE])}
    out = size_split([("a", 0.8, LARGE)], gts)
    assert out["large"].ap50 == pytest.approx(1.0)
    assert out["small"].ap50 == 0.0 and out["small"].recall == 0.0
    assert out["all"].recall == pytest.approx(0.5)


def test_false_positive_counts_only_in_its_own_range():
    gts = {"a": LARGE[None]}
    fp_small = ("a", 0.95, np.array([400.0, 400.0, 420.0, 420.0]))
    out = size_split([fp_small, ("a", 0.5, LARGE)], gts)
    assert out["large"].ap50 == pytest.approx(1.0)                 # the small FP is outside the range
    assert out["all"].ap50 < 1.0                                   # ...but ranked first overall


def test_detection_matched_to_ignored_gt_is_dropped():
    gts = {"a": np.stack([SMALL, LARGE])}
    out = ap50_in_range([("a", 0.99, LARGE), ("a", 0.5, SMALL)], gts, (0.0, 32.0 ** 2))
    assert out.ap50 == pytest.approx(1.0)                          # LARGE hit is neither TP nor FP here


def test_each_gt_matched_once():
    gts = {"a": SMALL[None]}
    out = ap50_in_range([("a", 0.9, SMALL), ("a", 0.8, SMALL + 0.5)], gts, (0.0, float("inf")))
    assert out.recall == pytest.approx(1.0) and out.ap50 == pytest.approx(1.0)   # duplicate after TP
