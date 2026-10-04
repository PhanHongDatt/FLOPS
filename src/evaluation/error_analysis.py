"""Diagnostic D1 — where do missed / false target-class detections come from? (ADR-014)

[ENGINEERING] Per image, compare predictions (conf >= 0.001, after NMS) with the
ground truth for one target class. No scientific claim of its own: it decomposes
the FN/FP counts the evaluator already reports.

Ground-truth outcome (first matching rule wins), at IoU >= ``iou_thr``:
  tp          a target-class prediction with conf >= ``conf_thr``
  confused_X  no such prediction, but a class-X prediction with conf >= conf_thr
  low_score   only target-class predictions below conf_thr
  missed      nothing at all at that location

Target-class prediction with conf >= conf_thr that matches no target GT:
  on_X        it sits on a class-X ground-truth box (IoU >= iou_thr): class confusion
  loc         it overlaps a target GT only loosely (0.1 <= IoU < iou_thr): localisation
  background  anything else

Per-GT "exists a match" counting, not Ultralytics' one-to-one greedy matching, so
totals can differ slightly from the confusion-matrix FP/FN; compare arms with each
other, not with those numbers.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

LOC_IOU = 0.1


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes, shape [len(a), len(b)]."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clip(0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clip(0).prod(-1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def decompose_image(
    gt_boxes: np.ndarray, gt_cls: np.ndarray,
    pred_boxes: np.ndarray, pred_cls: np.ndarray, pred_conf: np.ndarray,
    target: int, class_names: Sequence[str], conf_thr: float = 0.25, iou_thr: float = 0.5,
) -> tuple[Counter, Counter, list[float]]:
    """(GT outcomes, FP outcomes, best target score of every non-TP target GT) for one image."""
    gt_cls = np.asarray(gt_cls, dtype=int).reshape(-1)
    pred_cls = np.asarray(pred_cls, dtype=int).reshape(-1)
    pred_conf = np.asarray(pred_conf, dtype=float).reshape(-1)
    iou = box_iou(gt_boxes, pred_boxes)                       # [G, P]
    gt_out: Counter = Counter()
    fp_out: Counter = Counter()
    fn_scores: list[float] = []

    is_t = pred_cls == target
    strong = pred_conf >= conf_thr
    for g in np.flatnonzero(gt_cls == target):
        near = iou[g] >= iou_thr
        if (near & is_t & strong).any():
            gt_out["tp"] += 1
            continue
        t_scores = pred_conf[near & is_t]
        fn_scores.append(float(t_scores.max()) if t_scores.size else 0.0)
        other = near & ~is_t & strong
        if other.any():
            best = np.flatnonzero(other)[np.argmax(pred_conf[other])]
            gt_out[f"confused_{class_names[pred_cls[best]]}"] += 1
        elif t_scores.size:
            gt_out["low_score"] += 1
        else:
            gt_out["missed"] += 1

    tgt_gt = gt_cls == target
    for p in np.flatnonzero(is_t & strong):
        col = iou[:, p]
        if (col[tgt_gt] >= iou_thr).any():
            continue                                          # a true positive
        on_other = (col >= iou_thr) & ~tgt_gt
        if on_other.any():
            g = np.flatnonzero(on_other)[np.argmax(col[on_other])]
            fp_out[f"on_{class_names[gt_cls[g]]}"] += 1
        elif (col[tgt_gt] >= LOC_IOU).any():
            fp_out["loc"] += 1
        else:
            fp_out["background"] += 1
    return gt_out, fp_out, fn_scores


def yolo_labels_to_xyxy(lines: Sequence[str], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO ``cls cx cy w h`` (normalised) lines -> (xyxy pixel boxes, class ids)."""
    rows = [ln.split() for ln in lines if ln.strip()]
    if not rows:
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    arr = np.array([[float(v) for v in r[:5]] for r in rows])
    cls = arr[:, 0].astype(int)
    cx, cy, w, h = arr[:, 1] * width, arr[:, 2] * height, arr[:, 3] * width, arr[:, 4] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), cls
