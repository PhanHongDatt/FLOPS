"""Retrain only the classification branch (Detect.cv3) of a saved global model (ADR-038, ceiling test 1a).

Everything except ``model.<detect>.cv3.*`` is frozen through Ultralytics' ``freeze`` list; the frozen modules'
BatchNorm layers run in eval mode (``[YOLO-DOC]`` ultralytics 8.3.253 ``BaseTrainer._model_train``), so the neck
features the branch reads stay those of the checkpoint. The returned EMA still moves frozen tensors by float
rounding (EMA arithmetic, ~1e-7 relative on the smoke run); ``merge_cls_branch`` checks that bound and puts the
exact checkpoint values back outside cv3.
"""
from __future__ import annotations

import numpy as np

FROZEN_RTOL = 1e-5


def cls_branch_freeze_list(n_layers: int) -> list[str]:
    """Ultralytics ``freeze`` entries leaving only Detect.cv3 trainable (Detect = the last of ``n_layers``)."""
    det = n_layers - 1
    return [str(i) for i in range(det)] + [f"{det}.cv2", f"{det}.dfl"]


def merge_cls_branch(keys: list[str], before: list[np.ndarray], after: list[np.ndarray],
                     rtol: float = FROZEN_RTOL) -> tuple[list[np.ndarray], list[str]]:
    """``after`` inside cv3, ``before`` everywhere else; raises if a frozen tensor moved beyond rounding."""
    if not (len(keys) == len(before) == len(after)):
        raise ValueError(f"length mismatch: {len(keys)} keys, {len(before)} before, {len(after)} after")
    merged, changed, moved = [], [], []
    for k, b, a in zip(keys, before, after):
        if b.shape != a.shape:
            raise ValueError(f"{k}: shape {b.shape} vs {a.shape}")
        if ".cv3." in k:
            merged.append(a)
            if not np.array_equal(a, b):
                changed.append(k)
            continue
        if b.size and np.abs(b.astype(np.float64) - a).max() > rtol * (1.0 + np.abs(b).max()):
            moved.append(k)
        merged.append(b)
    if moved:
        raise ValueError(f"frozen parameters moved beyond rounding: {moved[:5]} (+{max(len(moved) - 5, 0)})")
    return merged, changed
