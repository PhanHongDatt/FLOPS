"""N1 (ADR-020): CCVR-style classifier calibration adapted to the YOLO Detect head.

[LITERATURE] Luo et al., "No Fear of Heterogeneity: Classifier Calibration for Federated Learning with
Non-IID Data", NeurIPS 2021 (arXiv:2106.05001): clients send per-class feature statistics, the server
samples virtual features from per-class Gaussians and re-trains the classifier.

[THESIS-HYPOTHESIS] detection adaptation (ADR-020):
- "classifier" = the last 1x1 class conv ``cv3[l][-1]`` of each Detect level l; its 128-d input is the feature;
- a ground-truth box contributes the feature at its centre cell on ONE level chosen by size
  (sqrt(w*h) < 64 px -> stride 8, < 128 -> stride 16, else stride 32); background = random cells outside boxes;
- only the rows of the target classes are re-fitted (sigmoid BCE, background = all-zero target).

Statistics are kept as sufficient sums (count, sum, sum of outer products) so that client contributions merge
exactly; no image or feature leaves a client — only these sums.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

BACKGROUND = -1
LEVEL_EDGES = (64.0, 128.0)          # sqrt(area) thresholds at 640 input -> level 0 / 1 / 2


def level_for_box(sqrt_area: float) -> int:
    for i, edge in enumerate(LEVEL_EDGES):
        if sqrt_area < edge:
            return i
    return len(LEVEL_EDGES)


@dataclass
class FeatureStats:
    """Sufficient statistics per (level, class) for d-dimensional features."""

    dim: int
    sums: dict[tuple[int, int], dict[str, np.ndarray | float]] = field(default_factory=dict)

    def add(self, level: int, cls: int, feats: np.ndarray) -> None:
        feats = np.asarray(feats, dtype=np.float64).reshape(-1, self.dim)
        if not len(feats):
            return
        s = self.sums.setdefault((level, cls), {"n": 0.0, "s1": np.zeros(self.dim), "s2": np.zeros((self.dim, self.dim))})
        s["n"] += len(feats)
        s["s1"] += feats.sum(0)
        s["s2"] += feats.T @ feats

    def merge(self, other: "FeatureStats") -> "FeatureStats":
        out = FeatureStats(self.dim, {k: {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
                                      for k, v in self.sums.items()})
        for k, v in other.sums.items():
            s = out.sums.setdefault(k, {"n": 0.0, "s1": np.zeros(self.dim), "s2": np.zeros((self.dim, self.dim))})
            s["n"] += v["n"]
            s["s1"] = s["s1"] + v["s1"]
            s["s2"] = s["s2"] + v["s2"]
        return out

    def gaussian(self, level: int, cls: int) -> tuple[np.ndarray, np.ndarray, float] | None:
        s = self.sums.get((level, cls))
        if s is None or s["n"] < 2:
            return None
        n = s["n"]
        mean = s["s1"] / n
        cov = (s["s2"] - n * np.outer(mean, mean)) / (n - 1)
        return mean, (cov + cov.T) / 2, n

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        out = {"dim": np.array(self.dim)}
        for (lvl, cls), v in self.sums.items():
            key = f"{lvl}_{cls}"
            out[f"n_{key}"] = np.array(v["n"])
            out[f"s1_{key}"] = v["s1"]
            out[f"s2_{key}"] = v["s2"]
        return out


def merge_all(stats: Iterable[FeatureStats]) -> FeatureStats:
    stats = list(stats)
    out = FeatureStats(stats[0].dim)
    for s in stats:
        out = out.merge(s)
    return out


def box_cells(boxes_xywh_px: np.ndarray, strides: Sequence[float], shape_hw: tuple[int, int]) -> list[tuple[int, int, int]]:
    """(level, gy, gx) of each box centre on its size-chosen level."""
    h, w = shape_hw
    out = []
    for cx, cy, bw, bh in np.asarray(boxes_xywh_px, dtype=float).reshape(-1, 4):
        lvl = level_for_box(float(np.sqrt(max(bw, 0) * max(bh, 0))))
        s = strides[lvl]
        gy = int(min(max(cy // s, 0), h // s - 1))
        gx = int(min(max(cx // s, 0), w // s - 1))
        out.append((lvl, gy, gx))
    return out


def background_mask(boxes_xywh_px: np.ndarray, stride: float, gh: int, gw: int) -> np.ndarray:
    """True for grid cells whose centre lies outside every box."""
    ys = (np.arange(gh) + 0.5) * stride
    xs = (np.arange(gw) + 0.5) * stride
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    mask = np.ones((gh, gw), dtype=bool)
    for cx, cy, bw, bh in np.asarray(boxes_xywh_px, dtype=float).reshape(-1, 4):
        inside = (np.abs(xx - cx) <= bw / 2) & (np.abs(yy - cy) <= bh / 2)
        mask &= ~inside
    return mask


@torch.no_grad()
def collect_stats(detection_model: torch.nn.Module, batches: Iterable[dict], target_ids: Sequence[int],
                  device: str | int = "cpu", bg_per_image: int = 8, seed: int = 0) -> FeatureStats:
    """Client side: per-level, per-class sufficient statistics of the class-conv input features."""
    rng = np.random.default_rng(seed)
    det = detection_model.model[-1]
    strides = [float(s) for s in det.stride]
    feats: dict[int, torch.Tensor] = {}
    hooks = [det.cv3[l][-1].register_forward_pre_hook(lambda m, inp, l=l: feats.__setitem__(l, inp[0].detach()))
             for l in range(len(det.cv3))]
    dim = int(det.cv3[0][-1].in_channels)
    stats = FeatureStats(dim)
    targets = set(int(t) for t in target_ids)
    detection_model.eval()
    try:
        for batch in batches:
            img = batch["img"].to(device).float() / 255.0
            detection_model(img)
            _, _, H, W = img.shape
            cls = batch["cls"].view(-1).cpu().numpy().astype(int)
            bidx = batch["batch_idx"].view(-1).cpu().numpy().astype(int)
            xywh = batch["bboxes"].cpu().numpy() * np.array([W, H, W, H])
            for b in range(img.shape[0]):
                sel = bidx == b
                boxes, labels = xywh[sel], cls[sel]
                for (lvl, gy, gx), c in zip(box_cells(boxes, strides, (H, W)), labels):
                    if c in targets:
                        stats.add(lvl, int(c), feats[lvl][b, :, gy, gx].cpu().numpy())
                for lvl, s in enumerate(strides):
                    gh, gw = feats[lvl].shape[2:]
                    ys, xs = np.nonzero(background_mask(boxes, s, gh, gw))
                    if len(ys):
                        pick = rng.choice(len(ys), size=min(bg_per_image, len(ys)), replace=False)
                        stats.add(lvl, BACKGROUND, feats[lvl][b, :, ys[pick], xs[pick]].T.cpu().numpy())
    finally:
        for h in hooks:
            h.remove()
    return stats


def calibrate(detection_model: torch.nn.Module, stats: FeatureStats, target_ids: Sequence[int],
              n_pos: int = 2000, n_bg: int = 8000, steps: int = 300, lr: float = 1e-3, seed: int = 0,
              eps: float = 1e-4) -> dict[int, dict[str, float]]:
    """Server side: re-fit the target rows of every level's class conv on virtual features."""
    g = torch.Generator().manual_seed(seed)
    det = detection_model.model[-1]
    ids = [int(t) for t in target_ids]
    report: dict[int, dict[str, float]] = {}
    for lvl in range(len(det.cv3)):
        conv = det.cv3[lvl][-1]
        xs, ys = [], []
        for j, c in enumerate(ids + [BACKGROUND]):
            gs = stats.gaussian(lvl, c)
            if gs is None:
                continue
            mean, cov, _ = gs
            n = n_bg if c == BACKGROUND else n_pos
            d = len(mean)
            chol = torch.linalg.cholesky(torch.tensor(cov, dtype=torch.float64) + eps * torch.eye(d, dtype=torch.float64))
            z = torch.randn(n, d, generator=g, dtype=torch.float64)
            xs.append(torch.tensor(mean, dtype=torch.float64) + z @ chol.T)
            y = torch.zeros(n, len(ids), dtype=torch.float64)
            if c != BACKGROUND:
                y[:, j] = 1.0
            ys.append(y)
        if not xs or not any(y.sum() > 0 for y in ys):
            report[lvl] = {"skipped": 1.0}
            continue
        x = torch.cat(xs).float()
        y = torch.cat(ys).float()
        w = conv.weight.data[ids, :, 0, 0].detach().clone().float().requires_grad_(True)
        b = conv.bias.data[ids].detach().clone().float().requires_grad_(True)
        opt = torch.optim.Adam([w, b], lr=lr)
        loss0 = None
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w.T + b, y)
            loss0 = float(loss.detach()) if loss0 is None else loss0
            loss.backward()
            opt.step()
        with torch.no_grad():
            conv.weight.data[ids, :, 0, 0] = w.to(conv.weight.dtype)
            conv.bias.data[ids] = b.to(conv.bias.dtype)
        report[lvl] = {"samples": float(len(x)), "loss_start": loss0, "loss_end": float(loss.detach())}
    return report
