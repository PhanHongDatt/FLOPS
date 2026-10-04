"""A6 — Equalized Focal Loss, and its control A6c (plain focal loss) (ADR-013).

[LITERATURE] Li et al., "Equalized Focal Loss for Dense Long-Tailed Object
Detection", CVPR 2022 (official code github.com/ModelTC/EOD,
``up/tasks/det/plugins/efl/models/losses/efl.py``). For class j:

    gamma_j = gamma_b + s * (1 - g_j)          focusing factor
    w_j     = gamma_j / gamma_b                weighting factor
    loss    = alpha_t * w_j * (1 - p_t)^gamma_j * CE

g_j = clamp(sum|grad_pos_j| / sum|grad_neg_j|, 0, 1), accumulated over training
from the gradient of the classification output (initial g_j = 1, i.e. plain focal).
Official defaults: gamma_b = 2, alpha = 0.25; s = 4 in the authors' YOLOX configs
(``configs/det/efl/efl_yolox_small.yaml``), the one-stage setting closest to YOLOv8.

Why it fits the A2b failure (ADR-006, s5/s6 results): A2b scaled the WHOLE negative
term of a locally missing class by rho and FP bus rose ~3x. EFL also raises the
focusing for a class whose positives are scarce, but focal modulation only
suppresses EASY negatives; confidently wrong scores (the FPs) keep a large loss,
and w_j > 1 even amplifies them.

[YOLO-DOC] The focal form with soft (task-aligned) targets follows ultralytics
8.3.253 ``utils/loss.py FocalLoss``: BCE * (1 - p_t)^gamma * alpha_t with
p_t = t*p + (1-t)*(1-p), without its reduction (v8DetectionLoss sums and divides by
target_scores_sum itself).

[THESIS-HYPOTHESIS] Federated adaptation: the official code all-reduces the
gradient statistics over GPUs; here each client accumulates its own statistics,
persisted across rounds in its run directory. For a class the client never sees,
g_j = 0, so it gets the maximum focusing gamma_b + s and weight (gamma_b + s)/gamma_b.
Pooling the statistics on the server is the untested alternative (ADR-013).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


class _FocalBCE(torch.nn.Module):
    """Drop-in for ``v8DetectionLoss.bce`` (elementwise). ``equalize=False`` is plain focal."""

    def __init__(self, nc: int, gamma: float, alpha: float, scale: float, equalize: bool) -> None:
        super().__init__()
        self.gamma, self.alpha, self.scale, self.equalize = float(gamma), float(alpha), float(scale), equalize
        self.pos_grad = torch.zeros(nc, dtype=torch.float64)
        self.neg_grad = torch.zeros(nc, dtype=torch.float64)
        self.pos_neg = torch.ones(nc, dtype=torch.float64)
        self.grad_scale = lambda: 1.0          # AMP GradScaler factor; set by FocalClassLoss

    def class_gamma(self) -> torch.Tensor:
        if not self.equalize:
            return torch.full_like(self.pos_neg, self.gamma)
        return self.gamma + self.scale * (1.0 - self.pos_neg)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        z = logits.float()                     # fp32 under AMP: (1 - p_t)^6 underflows in fp16
        t = target.float()
        ce = F.binary_cross_entropy_with_logits(z, t, reduction="none")
        p = torch.sigmoid(z)
        p_t = t * p + (1 - t) * (1 - p)
        gamma = self.class_gamma().to(device=z.device, dtype=z.dtype)
        weight = gamma / self.gamma
        alpha_t = t * self.alpha + (1 - t) * (1 - self.alpha)
        loss = ce * (1.0 - p_t).pow(gamma) * weight * alpha_t
        if self.equalize and z.requires_grad:
            t_detached = t.detach()
            z.register_hook(lambda g: self._collect(g, t_detached))
        return loss

    def _collect(self, grad: torch.Tensor, target: torch.Tensor) -> None:
        # The hook sees GradScaler-scaled gradients (as in federated/proximal.py); unscale so
        # batches before and after a scale change weigh the same.
        g = grad.detach().abs().double() / self.grad_scale()
        if not torch.isfinite(g).all():        # AMP overflow step: the optimizer skips it too
            return
        t = target.double()
        dims = tuple(range(g.dim() - 1))
        self.pos_grad += (g * t).sum(dims).cpu()
        self.neg_grad += (g * (1 - t)).sum(dims).cpu()
        self.pos_neg = torch.clamp(self.pos_grad / (self.neg_grad + 1e-10), 0.0, 1.0)

    def state(self) -> dict[str, list[float]]:
        return {"pos_grad": self.pos_grad.tolist(), "neg_grad": self.neg_grad.tolist()}

    def load_state(self, state: dict[str, list[float]]) -> None:
        self.pos_grad = torch.tensor(state["pos_grad"], dtype=torch.float64)
        self.neg_grad = torch.tensor(state["neg_grad"], dtype=torch.float64)
        self.pos_neg = torch.clamp(self.pos_grad / (self.neg_grad + 1e-10), 0.0, 1.0)


class FocalClassLoss:
    """Ultralytics callbacks for A6 (``equalize=True``) and A6c (``equalize=False``).

    ``state_path`` keeps a client's EFL gradient statistics across FL rounds
    (Flower recreates clients every round).
    """

    def __init__(self, equalize: bool, gamma: float = 2.0, alpha: float = 0.25, scale: float = 4.0,
                 state_path: Path | None = None) -> None:
        if gamma <= 0 or not 0 <= alpha <= 1 or scale < 0:
            raise ValueError(f"bad focal parameters gamma={gamma} alpha={alpha} scale={scale}")
        self.equalize, self.gamma, self.alpha, self.scale = equalize, gamma, alpha, scale
        self.state_path = Path(state_path) if state_path else None
        self._criterion: Any = None
        self._original: Any = None
        self.loss: _FocalBCE | None = None

    active = True

    def on_train_start(self, trainer: Any) -> None:
        model = trainer.model
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()
        crit = model.criterion
        if isinstance(crit.bce, _FocalBCE):          # re-entry
            return
        self.loss = _FocalBCE(crit.nc, self.gamma, self.alpha, self.scale, self.equalize)
        scaler = getattr(trainer, "scaler", None)
        if scaler is not None:
            self.loss.grad_scale = lambda: (float(scaler.get_scale())
                                            if getattr(scaler, "is_enabled", lambda: True)() else 1.0)
        if self.equalize and self.state_path and self.state_path.exists():
            self.loss.load_state(json.loads(self.state_path.read_text(encoding="utf-8")))
        self._criterion, self._original = crit, crit.bce
        crit.bce = self.loss

    def on_train_end(self, trainer: Any) -> None:
        if self.loss is not None and self.equalize and self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")      # atomic: a watchdog kill never leaves half a file
            tmp.write_text(json.dumps(self.loss.state()), encoding="utf-8")
            os.replace(tmp, self.state_path)
        if self._criterion is not None:
            self._criterion.bce = self._original
        self._criterion = self._original = None
