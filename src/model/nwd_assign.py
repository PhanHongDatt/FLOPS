"""M3 — mix the Normalized Gaussian Wasserstein Distance into TAL label assignment (docs/Solution.md §4, ADR-029).

[LITERATURE] Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection",
arXiv:2110.13389 (2021). A box (cx, cy, w, h) is modelled as a 2-D Gaussian; between two boxes

    W2^2 = (cx1 - cx2)^2 + (cy1 - cy2)^2 + ((w1 - w2) / 2)^2 + ((h1 - h2) / 2)^2
    NWD  = exp(-sqrt(W2^2) / C)

Unlike IoU it does not collapse to 0 for a few pixels of offset on a tiny box, so more anchors qualify as
positives for small objects. C is a dataset-dependent constant; here C = mean sqrt(w * h) of the client's
motorcycle boxes at the training resolution, computed from its own labels and logged, not the paper's value.

[YOLO-DOC] ultralytics 8.3.253 ``TaskAlignedAssigner.get_box_metrics`` uses ``iou_calculation`` (CIoU,
clamped at 0, xyxy pixels) both in ``align_metric = s^alpha * overlap^beta`` and for the normalised target
scores. The subclass replaces only that overlap by ``a * IoU + (1 - a) * NWD`` `[THESIS-HYPOTHESIS]`;
a = 1 gives the original assigner. Box/DFL losses are not changed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ultralytics.utils.tal import TaskAlignedAssigner


def nwd(box1: torch.Tensor, box2: torch.Tensor, c: float) -> torch.Tensor:
    """NWD of xyxy boxes [..., 4] (broadcast), in (0, 1]."""
    w1, h1 = box1[..., 2] - box1[..., 0], box1[..., 3] - box1[..., 1]
    w2, h2 = box2[..., 2] - box2[..., 0], box2[..., 3] - box2[..., 1]
    dx = (box1[..., 0] + box1[..., 2] - box2[..., 0] - box2[..., 2]) / 2
    dy = (box1[..., 1] + box1[..., 3] - box2[..., 1] - box2[..., 3]) / 2
    w2sq = dx ** 2 + dy ** 2 + ((w1 - w2) / 2) ** 2 + ((h1 - h2) / 2) ** 2
    return torch.exp(-torch.sqrt(w2sq.clamp(min=0)) / c)


class NWDTaskAlignedAssigner(TaskAlignedAssigner):
    def __init__(self, base: TaskAlignedAssigner, alpha_mix: float, c: float) -> None:
        super().__init__(topk=base.topk, num_classes=base.num_classes, alpha=base.alpha, beta=base.beta, eps=base.eps)
        if not 0 <= alpha_mix <= 1 or c <= 0:
            raise ValueError(f"need 0 <= alpha <= 1 and C > 0, got alpha={alpha_mix}, C={c}")
        self.alpha_mix, self.c = float(alpha_mix), float(c)

    def iou_calculation(self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        iou = super().iou_calculation(gt_bboxes, pd_bboxes)
        if self.alpha_mix == 1.0:
            return iou
        d = nwd(gt_bboxes, pd_bboxes, self.c).to(iou.dtype)
        return self.alpha_mix * iou + (1 - self.alpha_mix) * d


def mean_box_size(labels: list[dict], class_id: int, imgsz: int) -> float | None:
    """Mean sqrt(w * h) in pixels of ``class_id`` boxes after the long side is resized to ``imgsz``;
    each image counted once (repeat-factor sampling lists some images several times)."""
    sizes, seen = [], set()
    for lab in labels:
        if lab["im_file"] in seen:
            continue
        seen.add(lab["im_file"])
        h, w = lab["shape"]
        r = imgsz / max(h, w)
        for c, (_, _, bw, bh) in zip(lab["cls"].reshape(-1), lab["bboxes"]):
            if int(c) == class_id:
                sizes.append(np.sqrt(bw * w * r * bh * h * r))
    return float(np.mean(sizes)) if sizes else None


class NWDAssign:
    """Ultralytics callbacks: swap the criterion's assigner at train start, restore it at the end."""

    def __init__(self, alpha: float = 0.5, c: float | None = None, class_id: int = 3) -> None:
        if not 0 <= alpha <= 1:
            raise ValueError(f"nwd_assign.alpha must be in [0, 1], got {alpha}")
        self.alpha, self.c, self.class_id = float(alpha), c, int(class_id)
        self.c_used: float | None = None
        self._crit: Any = None
        self._original: Any = None

    def on_train_start(self, trainer: Any) -> None:
        model = trainer.model
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()
        crit = getattr(model.criterion, "criterion", model.criterion)   # under the teacher wrapper, if any
        c = self.c
        if c is None:
            c = mean_box_size(trainer.train_loader.dataset.labels, self.class_id, int(trainer.args.imgsz))
            if c is None:
                raise ValueError(f"nwd_assign: no class-{self.class_id} box to compute C from; set C explicitly")
        self.c_used = float(c)
        self._crit, self._original = crit, crit.assigner
        crit.assigner = NWDTaskAlignedAssigner(crit.assigner, self.alpha, self.c_used)

    def on_train_end(self, trainer: Any) -> None:
        if self._crit is not None:
            self._crit.assigner = self._original
        self._crit = self._original = None
