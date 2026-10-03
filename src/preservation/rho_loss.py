"""A2b — loss-level knowledge preservation (H2, ADR-006).

[THESIS-HYPOTHESIS] For a class a client has no positives for, ordinary BCE only
pushes that class's logits down at every anchor ("there is no bus here"),
erasing global knowledge. Scaling that class's BCE term by rho in [0, 1] limits
the pressure: rho = 1 is ordinary training, rho = 0 removes it.

[YOLO-DOC] ultralytics 8.3.253 ``v8DetectionLoss.bce`` is
``BCEWithLogitsLoss(reduction="none")`` applied to ``pred_scores`` of shape
[B, anchors, nc], summed and divided by ``target_scores_sum``; the criterion is
created lazily by ``DetectionModel.init_criterion()``. Wrapping ``bce`` with a
per-class weight vector therefore changes exactly the classification term of
the chosen classes and nothing else (box / DFL losses untouched).

Unlike A2a (parameter mask after training), this changes the trajectory of the
SHARED parameters too, which is why it is separable from server-side A3
(plan.md §6.3). Known limits (plan.md §6.1): weight decay still shrinks the
protected class-row weights slightly; EMA is unaffected in kind.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def class_loss_weights(class_names: Sequence[str], missing: Sequence[str], rho: float) -> list[float]:
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    unknown = [c for c in missing if c not in class_names]
    if unknown:
        raise ValueError(f"missing classes {unknown} not in {list(class_names)}")
    return [rho if c in missing else 1.0 for c in class_names]


class _WeightedBCE(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, weights: Sequence[float]) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("w", torch.tensor(list(weights), dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base(logits, target)
        return loss * self.w.to(device=loss.device, dtype=loss.dtype)


class RhoClassLoss:
    """Ultralytics callbacks: wrap the criterion's BCE at train start, restore at end."""

    def __init__(self, weights: Sequence[float]) -> None:
        self.weights = [float(w) for w in weights]
        self._criterion: Any = None
        self._original: Any = None

    @property
    def active(self) -> bool:
        return any(w != 1.0 for w in self.weights)

    def on_train_start(self, trainer: Any) -> None:
        if not self.active:
            return
        model = trainer.model
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()
        crit = model.criterion
        if isinstance(crit.bce, _WeightedBCE):          # already wrapped (re-entry)
            return
        self._criterion, self._original = crit, crit.bce
        crit.bce = _WeightedBCE(crit.bce, self.weights)

    def on_train_end(self, trainer: Any) -> None:
        if self._criterion is not None:
            self._criterion.bce = self._original
        self._criterion = self._original = None
