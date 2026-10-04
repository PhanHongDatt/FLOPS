"""Classification-loss variants selectable per client (A5, A6, A6c).

A spec is a plain dict so it can travel through the client's train kwargs:
  {"kind": "ntd",   "beta": 1.0, "tau": 1.0}                         A5  (ADR-012)
  {"kind": "efl",   "gamma": 2.0, "alpha": 0.25, "scale": 4.0,
   "state_path": ".../efl_state.json"}                                A6  (ADR-013)
  {"kind": "focal", "gamma": 2.0, "alpha": 0.25}                      A6c (ADR-013)
Each returned object exposes ``on_train_start`` / ``on_train_end`` Ultralytics callbacks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.preservation.efl_loss import FocalClassLoss
from src.preservation.ntd_loss import NotTrueDistillation

LOSS_KINDS = ("ntd", "efl", "focal")


def build_loss_plugin(spec: dict[str, Any]) -> Any:
    kind = spec.get("kind")
    if kind == "ntd":
        return NotTrueDistillation(beta=float(spec.get("beta", 1.0)), tau=float(spec.get("tau", 1.0)))
    if kind in ("efl", "focal"):
        state = spec.get("state_path")
        return FocalClassLoss(
            equalize=kind == "efl",
            gamma=float(spec.get("gamma", 2.0)),
            alpha=float(spec.get("alpha", 0.25)),
            scale=float(spec.get("scale", 4.0)),
            state_path=Path(state) if state else None,
        )
    raise ValueError(f"unknown classification-loss kind {kind!r}; expected one of {LOSS_KINDS}")
