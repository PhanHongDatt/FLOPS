"""D1 — FN / FP decomposition for one target class (ADR-014)."""
from __future__ import annotations

import numpy as np

from src.evaluation.error_analysis import box_iou, decompose_image, yolo_labels_to_xyxy

NAMES = ("car", "bus", "truck", "motorcycle")
BUS, TRUCK, CAR = 1, 2, 0


def _run(gt, gc, pb, pc, pf):
    return decompose_image(np.array(gt, float), np.array(gc), np.array(pb, float).reshape(-1, 4),
                           np.array(pc), np.array(pf), BUS, NAMES)


def test_iou():
    assert box_iou([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0] == 1.0
    assert box_iou([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] == 50 / 150
    assert box_iou(np.zeros((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1)


def test_gt_outcomes_in_priority_order():
    gt = [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10], [60, 0, 70, 10]]
    gc = [BUS, BUS, BUS, BUS]
    preds = [
        ([0, 0, 10, 10], BUS, 0.9),       # gt0: tp
        ([20, 0, 30, 10], TRUCK, 0.6),    # gt1: confused as truck (beats the weak bus box below)
        ([20, 0, 30, 10], BUS, 0.1),
        ([40, 0, 50, 10], BUS, 0.05),     # gt2: low score
    ]                                     # gt3: nothing -> missed
    pb, pc, pf = zip(*preds)
    g, f, s = _run(gt, gc, pb, pc, pf)
    assert g == {"tp": 1, "confused_truck": 1, "low_score": 1, "missed": 1}
    assert sorted(s) == [0.0, 0.05, 0.1]
    assert not f


def test_fp_outcomes():
    gt = [[0, 0, 10, 10], [20, 0, 30, 10]]
    gc = [TRUCK, BUS]
    preds = [
        ([0, 0, 10, 10], BUS, 0.8),       # on a truck
        ([24, 0, 34, 10], BUS, 0.7),      # IoU 6/14 with the bus: localisation
        ([80, 80, 90, 90], BUS, 0.5),     # background
        ([80, 80, 90, 90], BUS, 0.1),     # below threshold: ignored
        ([20, 0, 30, 10], BUS, 0.9),      # the true positive
    ]
    pb, pc, pf = zip(*preds)
    g, f, _ = _run(gt, gc, pb, pc, pf)
    assert g == {"tp": 1}
    assert f == {"on_truck": 1, "loc": 1, "background": 1}


def test_empty_image():
    g, f, s = decompose_image(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 4)), np.zeros(0), np.zeros(0),
                              BUS, NAMES)
    assert not g and not f and s == []


def test_yolo_label_conversion():
    boxes, cls = yolo_labels_to_xyxy(["1 0.5 0.5 0.2 0.4", ""], width=100, height=50)
    assert cls.tolist() == [1]
    assert np.allclose(boxes, [[40, 15, 60, 35]])
    assert yolo_labels_to_xyxy([], 10, 10)[0].shape == (0, 4)


def test_find_runs_reads_plain_dirs_and_session_zips(tmp_path):
    import json
    import zipfile

    from scripts.analyze_errors import find_runs

    def make_run(root, name, seed):
        run = root / "artifacts" / "runs" / name
        (run / "checkpoint").mkdir(parents=True)
        (run / "environment.json").write_text(json.dumps({"exp_id": "A0", "seed": seed, "partition_id": "p"}))
        np.savez(run / "checkpoint" / "global_round_030.npz", np.zeros(2))
        return run

    make_run(tmp_path / "live", "A0_seed1", 1)
    zsrc = tmp_path / "zsrc"
    make_run(zsrc, "A0_seed2", 2)
    (tmp_path / "input" / "s6a").mkdir(parents=True)
    with zipfile.ZipFile(tmp_path / "input" / "s6a" / "flops_results.zip", "w") as z:
        for f in zsrc.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(zsrc).as_posix())
    runs = find_runs([tmp_path / "live", tmp_path / "input"], 30, tmp_path / "x")
    assert [env["seed"] for env, _ in runs] == [1, 2]
    assert all(p.exists() for _, p in runs)
    assert find_runs([tmp_path / "live"], 29, tmp_path / "x") == []
