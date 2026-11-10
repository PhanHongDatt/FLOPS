"""AP50 per object-size range (docs/Solution.md §6; ADR-029 rule 2, ADR-032).

Size ranges follow the COCO convention `[LITERATURE]` (Lin et al., ECCV 2014; pycocotools ``areaRng``): box
area in ORIGINAL image pixels, small < 32², medium 32²-96², large >= 96². Matching follows pycocotools for
one area range: ground truths outside the range are "ignore"; a detection matched to an ignored ground truth
is dropped, and an unmatched detection whose own area lies outside the range is dropped too. Greedy matching
by score, IoU >= 0.5, each ground truth used once, preferring a non-ignored one. AP = 101-point interpolated
precision (pycocotools). Detections of one class only; class-aware NMS has already run in the predictor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AREA_RANGES: dict[str, tuple[float, float]] = {
    "all": (0.0, float("inf")),
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, float("inf")),
}
IOU_THR = 0.5


@dataclass(frozen=True)
class SizeAP:
    ap50: float
    n_gt: int
    recall: float


def _area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    return inter / np.maximum(_area(box[None])[0] + _area(boxes) - inter, 1e-12)


def ap50_in_range(dets: list[tuple[str, float, np.ndarray]], gts: dict[str, np.ndarray],
                  rng: tuple[float, float]) -> SizeAP:
    """``dets``: (image id, score, xyxy) of one class; ``gts``: image id -> [N, 4] xyxy of that class."""
    lo, hi = rng
    ignore = {k: ~((_area(v) >= lo) & (_area(v) < hi)) if len(v) else np.zeros(0, bool) for k, v in gts.items()}
    n_gt = int(sum((~ig).sum() for ig in ignore.values()))
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    flags: list[int] = []                                    # 1 TP, 0 FP, in score order (dropped omitted)
    for img, _, box in sorted(dets, key=lambda d: -d[1]):
        g = gts.get(img, np.zeros((0, 4)))
        match = -1
        if len(g):
            ious = _iou(box, g)
            ig = ignore[img]
            for want_ignored in (False, True):               # a non-ignored ground truth first
                cand = np.where((~used[img]) & (ig == want_ignored) & (ious >= IOU_THR))[0]
                if len(cand):
                    match = int(cand[np.argmax(ious[cand])])
                    break
        if match >= 0:
            used[img][match] = True
            if not ignore[img][match]:
                flags.append(1)
            continue
        a = _area(box[None])[0]
        if lo <= a < hi:
            flags.append(0)
    if n_gt == 0:
        return SizeAP(float("nan"), 0, float("nan"))
    tp = np.cumsum(np.array(flags, float)) if flags else np.zeros(0)
    fp = np.cumsum(1 - np.array(flags, float)) if flags else np.zeros(0)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec   # monotone envelope
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, points, side="left")
    interp = np.array([prec[i] if i < len(prec) else 0.0 for i in idx])
    return SizeAP(float(interp.mean()), n_gt, float(rec[-1]) if len(rec) else 0.0)


def size_split(dets: list[tuple[str, float, np.ndarray]], gts: dict[str, np.ndarray]) -> dict[str, SizeAP]:
    return {name: ap50_in_range(dets, gts, rng) for name, rng in AREA_RANGES.items()}
