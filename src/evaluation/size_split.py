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


@dataclass(frozen=True)
class Matched:
    """Per-detection outcome for one class and size range: image index, score, flag (1 TP, 0 FP), and the
    number of in-range ground truths per image. Matching is image-local, so images can be resampled."""
    img: np.ndarray
    score: np.ndarray
    flag: np.ndarray
    n_gt: np.ndarray            # [n_images]


def match_flags(dets: list[tuple[str, float, np.ndarray]], gts: dict[str, np.ndarray],
                rng: tuple[float, float], images: list[str] | None = None) -> Matched:
    """``dets``: (image id, score, xyxy) of one class; ``gts``: image id -> [N, 4] xyxy of that class.
    Dropped detections (matched to an ignored ground truth, or unmatched and out of range) are omitted."""
    lo, hi = rng
    images = sorted(gts) if images is None else images
    index = {k: i for i, k in enumerate(images)}
    ignore = {k: ~((_area(v) >= lo) & (_area(v) < hi)) if len(v) else np.zeros(0, bool) for k, v in gts.items()}
    n_gt = np.array([int((~ignore[k]).sum()) if k in ignore else 0 for k in images])
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    out_img, out_score, out_flag = [], [], []
    for img, score, box in sorted(dets, key=lambda d: -d[1]):
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
            flag = None if ignore[img][match] else 1
        else:
            flag = 0 if lo <= _area(box[None])[0] < hi else None
        if flag is not None:
            out_img.append(index[img])
            out_score.append(score)
            out_flag.append(flag)
    return Matched(np.array(out_img, int), np.array(out_score, float), np.array(out_flag, int), n_gt)


def ap_from_flags(score: np.ndarray, flag: np.ndarray, n_gt: int) -> tuple[float, float]:
    """(AP50 101-point interpolated, final recall) from per-detection scores and TP flags."""
    if n_gt == 0:
        return float("nan"), float("nan")
    order = np.argsort(-score, kind="stable")
    f = flag[order].astype(float)
    tp, fp = np.cumsum(f), np.cumsum(1 - f)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-12)
    if not len(prec):
        return 0.0, 0.0
    prec = np.maximum.accumulate(prec[::-1])[::-1]           # monotone envelope
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    interp = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)
    return float(np.mean(interp)), float(rec[-1])


def ap50_in_range(dets: list[tuple[str, float, np.ndarray]], gts: dict[str, np.ndarray],
                  rng: tuple[float, float]) -> SizeAP:
    m = match_flags(dets, gts, rng)
    ap, rec = ap_from_flags(m.score, m.flag, int(m.n_gt.sum()))
    return SizeAP(ap, int(m.n_gt.sum()), rec)


def size_split(dets: list[tuple[str, float, np.ndarray]], gts: dict[str, np.ndarray]) -> dict[str, SizeAP]:
    return {name: ap50_in_range(dets, gts, rng) for name, rng in AREA_RANGES.items()}


def bootstrap_ap(m: Matched, weights: np.ndarray) -> float:
    """AP50 on a bootstrap resample: ``weights[i]`` = how many times image i was drawn."""
    w = weights[m.img]
    keep = w > 0
    score, flag = np.repeat(m.score[keep], w[keep]), np.repeat(m.flag[keep], w[keep])
    return ap_from_flags(score, flag, int((m.n_gt * weights).sum()))[0]


def paired_bootstrap(a: Matched, b: Matched, n: int = 1000, seed: int = 0) -> dict[str, float]:
    """95 % percentile interval of AP50(b) - AP50(a), resampling the SAME images for both models
    (Efron & Tibshirani 1993, percentile bootstrap). ``a`` and ``b`` must index the same image list."""
    if len(a.n_gt) != len(b.n_gt) or not np.array_equal(a.n_gt, b.n_gt):
        raise ValueError("paired bootstrap needs both models matched on the same images and ground truths")
    rng = np.random.default_rng(seed)
    n_img = len(a.n_gt)
    diffs = np.empty(n)
    for k in range(n):
        weights = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)
        diffs[k] = bootstrap_ap(b, weights) - bootstrap_ap(a, weights)
    point = ap_from_flags(b.score, b.flag, int(b.n_gt.sum()))[0] - ap_from_flags(a.score, a.flag, int(a.n_gt.sum()))[0]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_boot": n}
