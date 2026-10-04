"""A5 — not-true distillation from the received global model (ADR-012).

[LITERATURE] Lee et al., "Preservation of the Global Knowledge by Not-True
Distillation in Federated Learning", NeurIPS 2022 (FedNTD; official code
github.com/Lee-Gihun/FedNTD, ``algorithms/fedntd/criterion.py``):

    L = CE(q, y) + beta * tau^2 * KL( softmax_nt(z_g / tau) || softmax_nt(z_l / tau) )

where ``_nt`` keeps only the not-true classes of each sample and z_g are the
logits of the global model received at the start of the round (frozen).
Official defaults (``config/fedntd.json``): tau = 1, beta = 1.

[THESIS-HYPOTHESIS] Adaptation to the YOLOv8 sigmoid head. Each (anchor, class)
score is an independent Bernoulli, so the softmax-over-not-true-classes KL becomes
a per-entry Bernoulli KL, applied where the task-aligned target is exactly 0 (the
class is "not true" at that anchor; background anchors have every class not true):

    loss_entry = BCE(z, t) + beta * tau^2 * [t == 0] * KL( Ber(s(z_g / tau)) || Ber(s(z / tau)) )

The KD entries go through the same ``sum / target_scores_sum`` normalisation as the
ordinary BCE, so beta = 1 gives the distillation the same scale as the hard-negative
term it accompanies. For a class a client never sees (bus on C0/C1) this keeps
the global model's bus scores as a soft target instead of letting the hard target 0
push them down everywhere — unlike A2b, the negative signal is not removed.

Installed through Ultralytics callbacks like A2b (rho_loss.py): ``on_train_start``
snapshots the trainer's model (= the received global parameters, as in
federated/proximal.py) as the frozen teacher, wraps the criterion so the teacher
scores the same batch, and wraps ``criterion.bce``; ``on_train_end`` restores both.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F


def bernoulli_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, tau: float) -> torch.Tensor:
    """Elementwise KL(Ber(s(t / tau)) || Ber(s(s / tau))) * tau^2; zero where the two agree."""
    zs, zt = student_logits / tau, teacher_logits / tau
    pt = torch.sigmoid(zt)
    cross = F.binary_cross_entropy_with_logits(zs, pt, reduction="none")
    entropy = F.binary_cross_entropy_with_logits(zt, pt, reduction="none")
    return (cross - entropy) * tau * tau


class _NTDBCE(torch.nn.Module):
    """Drop-in for ``v8DetectionLoss.bce``: ordinary BCE + not-true distillation."""

    def __init__(self, base: torch.nn.Module, beta: float, tau: float) -> None:
        super().__init__()
        self.base = base
        self.beta = float(beta)
        self.tau = float(tau)
        self.teacher_logits: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base(logits, target)
        if self.teacher_logits is None:
            return loss
        if self.teacher_logits.shape != logits.shape:
            raise RuntimeError(f"teacher scores {tuple(self.teacher_logits.shape)} do not match "
                               f"student scores {tuple(logits.shape)}")
        not_true = (target == 0).to(loss.dtype)
        kd = bernoulli_kl(logits.float(), self.teacher_logits.float(), self.tau).to(loss.dtype)
        return loss + self.beta * not_true * kd


def head_scores(feats: list[torch.Tensor], no: int, reg_max: int, nc: int) -> torch.Tensor:
    """Class logits [B, anchors, nc] in the exact layout ``v8DetectionLoss`` uses."""
    b = feats[0].shape[0]
    _, scores = torch.cat([xi.view(b, no, -1) for xi in feats], 2).split((reg_max * 4, nc), 1)
    return scores.permute(0, 2, 1).contiguous()


class _TeacherCriterion:
    """Wraps the detection criterion: the teacher scores each batch before the loss runs."""

    def __init__(self, criterion: Any, teacher: torch.nn.Module, bce: _NTDBCE) -> None:
        self.criterion = criterion
        self.teacher = teacher
        self.bce = bce

    def __getattr__(self, name: str) -> Any:     # anything else (hyp, nc, ...) from the real criterion
        if name == "criterion":                   # not set yet (copy/pickle): avoid infinite recursion
            raise AttributeError(name)
        return getattr(self.criterion, name)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> Any:
        crit = self.criterion
        with torch.no_grad():
            out = self.teacher(batch["img"])
            feats = out[1] if isinstance(out, tuple) else out
            self.bce.teacher_logits = head_scores(feats, crit.no, crit.reg_max, crit.nc).detach()
        try:
            return crit(preds, batch)
        finally:
            self.bce.teacher_logits = None


class NotTrueDistillation:
    """Ultralytics callbacks for A5. beta = 0 disables it (ordinary training)."""

    def __init__(self, beta: float = 1.0, tau: float = 1.0) -> None:
        if beta < 0 or tau <= 0:
            raise ValueError(f"need beta >= 0 and tau > 0, got beta={beta}, tau={tau}")
        self.beta, self.tau = float(beta), float(tau)
        self._model: Any = None
        self._criterion: Any = None
        self._original_bce: Any = None

    @property
    def active(self) -> bool:
        return self.beta > 0

    def on_train_start(self, trainer: Any) -> None:
        if not self.active:
            return
        model = trainer.model
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()
        if isinstance(model.criterion, _TeacherCriterion):   # re-entry
            return
        crit = model.criterion
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        bce = _NTDBCE(crit.bce, self.beta, self.tau)
        self._model, self._criterion, self._original_bce = model, crit, crit.bce
        crit.bce = bce
        model.criterion = _TeacherCriterion(crit, teacher, bce)

    def on_train_end(self, trainer: Any) -> None:
        if self._criterion is not None:
            self._criterion.bce = self._original_bce
            self._model.criterion = self._criterion
        self._model = self._criterion = self._original_bce = None
