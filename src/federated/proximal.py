"""FedProx proximal term for Ultralytics local training.

[LITERATURE] Li et al., 2020, MLSys (arXiv:1812.06127): client k minimises
    h_k(w) = F_k(w) + (mu / 2) * ||w - w^t||^2
so its gradient gains mu * (w - w^t). Ultralytics' trainer has no hook between
loss and backward, so the term is added as a per-parameter gradient hook on the
trainer's model, which is equivalent to adding it to the loss. [ENGINEERING]

* Under AMP the hook receives gradients multiplied by the GradScaler scale, so
  the proximal part is multiplied by the same scale.
* The hook fires on every backward; with gradient accumulation the proximal part
  accumulates like the loss gradient of each batch.
* w^t is snapshotted at ``on_train_start``, after Ultralytics built its model from
  the received global parameters.

Used through Ultralytics callbacks: ``on_train_start`` registers the hooks,
``on_train_end`` removes them.
"""
from __future__ import annotations

from typing import Any


class ProximalTerm:
    def __init__(self, mu: float) -> None:
        if mu < 0:
            raise ValueError(f"proximal mu must be >= 0, got {mu}")
        self.mu = float(mu)
        self._handles: list[Any] = []

    @property
    def active(self) -> bool:
        return self.mu > 0

    def on_train_start(self, trainer: Any) -> None:
        if not self.active:
            return
        scaler = getattr(trainer, "scaler", None)

        def scale() -> float:
            if scaler is None:
                return 1.0
            enabled = getattr(scaler, "is_enabled", lambda: True)()
            return float(scaler.get_scale()) if enabled else 1.0

        for p in trainer.model.parameters():
            if not p.requires_grad:
                continue
            anchor = p.detach().clone()

            def hook(grad, p=p, anchor=anchor):
                return grad + scale() * self.mu * (p.detach() - anchor)

            self._handles.append(p.register_hook(hook))

    def on_train_end(self, trainer: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
