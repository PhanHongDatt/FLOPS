"""Degradation-localisation probe (ADR-036): where does the missing-class damage sit?

Pure functions (numpy / torch, no Ultralytics): box sampling, cell lookup, image split, a fixed linear probe,
metrics, paired image bootstrap and the pre-registered decision rule. Feature extraction lives in
``scripts/extract_probe_features.py``; the analysis driver in ``scripts/probe_degradation.py``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

PROBE_CLASSES = ("car", "bus", "truck")      # target ids 0, 1, 2 (src.data.bdd100k.TARGET_CLASSES order)
STRIDES = (8, 16, 32)                         # P3 / P4 / P5
LEVEL_EDGES = (64.0, 192.0)                   # sqrt(area) in input pixels: < 64 -> P3, < 192 -> P4, else P5
CAR_KEEP_PER_MILLE = 78                       # ~8,000 of 102,506 val car boxes, model-independent
L2 = 1e-4
THRESHOLD = 0.03


def _unit(key: str) -> float:
    """Deterministic value in [0, 1) from a string (no Python hash randomisation)."""
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 2**32


def keep_box(image: str, index: int, cls: int) -> bool:
    """Bus and truck boxes are all kept; car boxes are sampled by a seeded hash of (image, box index)."""
    if cls not in (0, 1, 2):
        return False
    return cls != 0 or _unit(f"car|{image}|{index}") * 1000 < CAR_KEEP_PER_MILLE


def image_half(image: str, seed: int = 0) -> int:
    """0 = probe-train half, 1 = evaluation half; split by image, never by box."""
    return int(_unit(f"{seed}|{image}") >= 0.5)


def primary_level(box_xyxy_input: np.ndarray) -> np.ndarray:
    """Primary level (0, 1, 2) per box from sqrt(area) in letterboxed input pixels."""
    w = np.clip(box_xyxy_input[:, 2] - box_xyxy_input[:, 0], 0, None)
    h = np.clip(box_xyxy_input[:, 3] - box_xyxy_input[:, 1], 0, None)
    return np.digitize(np.sqrt(w * h), LEVEL_EDGES)


def centre_cells(box_xyxy_input: np.ndarray, grid_hw: list[tuple[int, int]]) -> np.ndarray:
    """[N, 3, 2] (row, col) of the cell holding each box centre on every level, clipped to the grid."""
    cx = (box_xyxy_input[:, 0] + box_xyxy_input[:, 2]) / 2
    cy = (box_xyxy_input[:, 1] + box_xyxy_input[:, 3]) / 2
    out = np.zeros((len(cx), len(STRIDES), 2), dtype=np.int64)
    for lvl, (s, (gh, gw)) in enumerate(zip(STRIDES, grid_hw)):
        out[:, lvl, 0] = np.clip(np.floor(cy / s), 0, gh - 1)
        out[:, lvl, 1] = np.clip(np.floor(cx / s), 0, gw - 1)
    return out


def balanced_index(y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Indices with every class subsampled to the smallest class count (deterministic)."""
    rng = np.random.default_rng(seed)
    n = min(int((y == c).sum()) for c in np.unique(y))
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), n, replace=False) for c in np.unique(y)]))


@dataclass(frozen=True)
class Probe:
    mean: np.ndarray
    std: np.ndarray
    weight: np.ndarray    # [D, K]
    bias: np.ndarray      # [K]

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x.astype(np.float32) - self.mean) / self.std
        return np.argmax(z @ self.weight + self.bias, axis=1)


def fit_probe(x: np.ndarray, y: np.ndarray, n_classes: int = 3, l2: float = L2, max_iter: int = 500) -> Probe:
    """Multinomial logistic regression, standardised inputs, fixed L2, full-batch LBFGS (no tuning)."""
    import torch

    x = x.astype(np.float32)
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt = torch.from_numpy((x - mean) / std)
    yt = torch.from_numpy(y.astype(np.int64))
    torch.manual_seed(0)
    w = torch.zeros(x.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(xt @ w + b, yt) + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return Probe(mean, std, w.detach().numpy(), b.detach().numpy())


def confusion(y: np.ndarray, pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """[true, predicted] counts."""
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(m, (y, pred), 1)
    return m


def recall_metrics(y: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """Per-class recall (bus = class 1) and macro accuracy from per-box correctness."""
    rec = [float(correct[y == c].mean()) if (y == c).any() else float("nan") for c in range(len(PROBE_CLASSES))]
    return {"bus_recall": rec[1], "macro_acc": float(np.nanmean(rec)),
            **{f"recall_{n}": r for n, r in zip(PROBE_CLASSES, rec)}}


def paired_image_bootstrap(images: np.ndarray, value_a: np.ndarray, value_b: np.ndarray, mask: np.ndarray,
                           n: int = 1000, seed: int = 0) -> dict[str, float]:
    """CI of mean(value_b) − mean(value_a) over boxes in ``mask``, resampling images (same draw for a and b)."""
    uniq, inv = np.unique(images, return_inverse=True)
    sa = np.bincount(inv, weights=value_a * mask, minlength=len(uniq))
    sb = np.bincount(inv, weights=value_b * mask, minlength=len(uniq))
    cnt = np.bincount(inv, weights=mask.astype(float), minlength=len(uniq))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n)
    for i in range(n):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        c = (w * cnt).sum()
        deltas[i] = ((w * sb).sum() - (w * sa).sum()) / c if c else np.nan
    delta = (sb.sum() - sa.sum()) / cnt.sum()
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return {"delta": float(delta), "ci_low": float(lo), "ci_high": float(hi)}


def meets(r: dict[str, float], threshold: float = THRESHOLD) -> bool:
    """Δ ≥ threshold and the 95 % CI excludes 0 (Δ = ideal − missing)."""
    return r["delta"] >= threshold and r["ci_low"] > 0


def decide(per_model: dict[str, dict[str, dict[str, float]]]) -> str:
    """ADR-036 rule. ``per_model[model][depth]`` with depths 'neck', 'cv3_1' (probe bus recall) and 'readout'
    (mean bus score); every model must agree, otherwise the outcome is reported as not replicated."""
    def outcome(d: dict[str, dict[str, float]]) -> str:
        if meets(d["neck"]):
            return "features degraded"
        if meets(d["cv3_1"]):
            return "head body degraded"
        if meets(d["readout"]):
            return "readout shifted"
        return "not a classification-separability effect"

    outcomes = {m: outcome(d) for m, d in per_model.items()}
    first = next(iter(outcomes.values()))
    if all(o == first for o in outcomes.values()):
        return first
    return "not replicated: " + "; ".join(f"{m}: {o}" for m, o in outcomes.items())
