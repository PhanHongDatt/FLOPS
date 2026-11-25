"""ADR-040 (descriptive): F1 per class at a per-class threshold tuned on the other half of the val images.

Two-fold cross-fit by image index (even / odd): the threshold that maximises F1 on one half is applied to the
other, so no image is scored with a threshold tuned on itself. Compared with the fixed conf 0.25.

Usage: python scripts/adr040_class_thresholds.py DUMP.npz [DUMP.npz ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CLASSES = ("car", "bus", "truck", "motorcycle")


def _counts(score: np.ndarray, flag: np.ndarray, n_gt: int, thr: float) -> tuple[int, int, int]:
    keep = score >= thr
    tp = int(flag[keep].sum())
    return tp, int(keep.sum()) - tp, n_gt - tp


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _best_thr(score: np.ndarray, flag: np.ndarray, n_gt: int) -> float:
    o = np.argsort(-score)
    s, f = score[o], flag[o]
    tp = np.cumsum(f)
    fp = np.cumsum(1 - f)
    f1 = 2 * tp / (tp + fp + n_gt)
    return float(s[int(f1.argmax())])


def cross_fit(path: Path, cls: str) -> dict[str, float]:
    p = np.load(path)
    img, score, flag, n_img = p[f"{cls}/all/img"], p[f"{cls}/all/score"], p[f"{cls}/all/flag"], p[f"{cls}/all/n_gt"]
    folds = [np.arange(len(n_img)) % 2 == k for k in (0, 1)]
    tot = np.zeros(3, int)
    thrs = []
    for k in (0, 1):
        tune, test = folds[1 - k], folds[k]
        on_tune, on_test = tune[img], test[img]
        thr = _best_thr(score[on_tune], flag[on_tune], int(n_img[tune].sum()))
        thrs.append(thr)
        tot += _counts(score[on_test], flag[on_test], int(n_img[test].sum()), thr)
    fixed = _counts(score, flag, int(n_img.sum()), 0.25)
    return {"F1@0.25": _f1(*fixed), "R@0.25": fixed[0] / max(fixed[0] + fixed[2], 1),
            "F1@cls": _f1(*tot), "R@cls": tot[0] / max(tot[0] + tot[2], 1), "thr": float(np.mean(thrs))}


def main() -> None:
    print("run | class | F1@0.25 | R@0.25 | F1@class-thr | R@class-thr | thr")
    for a in sys.argv[1:]:
        for c in CLASSES:
            r = cross_fit(Path(a), c)
            print(f"{Path(a).stem[:60]} | {c} | {r['F1@0.25']:.3f} | {r['R@0.25']:.3f} | "
                  f"{r['F1@cls']:.3f} | {r['R@cls']:.3f} | {r['thr']:.3f}")


if __name__ == "__main__":
    main()
