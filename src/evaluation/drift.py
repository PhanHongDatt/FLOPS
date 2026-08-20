"""Parameter drift analysis for G4 and F3.

Section 8 (F3) and Section 13 of CLAUDE.md require parameter/update evidence
BEYOND L2 norm alone. This module computes:
  - Δθ = θ_local - θ_global by module group (backbone/neck/detect_cls/detect_reg)
  - L2 norm per group
  - cosine similarity per group (flattened)
  - Direction (angle proxy)

A Missing-Class claim requires parameter evidence + prediction evidence.
This module handles the parameter half only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.model.parameter_map import classify_parameter


@dataclass(frozen=True)
class GroupDrift:
    module_group: str
    l2_norm: float
    cosine_similarity: float
    num_params: int
    total_numel: int


def _flatten_group(
    named_deltas: list[tuple[str, np.ndarray]],
    group: str,
) -> np.ndarray:
    parts = [d.ravel() for name, d in named_deltas if classify_parameter(name) == group]
    if not parts:
        return np.zeros(0)
    return np.concatenate(parts)


def compute_delta(
    local_state: dict[str, np.ndarray],
    global_state: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    """Δθ = θ_local - θ_global per named parameter (keys must match)."""
    common = local_state.keys() & global_state.keys()
    if len(common) != len(local_state) or len(common) != len(global_state):
        raise ValueError("Local and global state_dict keys must match exactly.")
    return [(k, local_state[k] - global_state[k]) for k in sorted(common)]


def group_drift(
    deltas: list[tuple[str, np.ndarray]],
    reference: list[tuple[str, np.ndarray]] | None = None,
) -> list[GroupDrift]:
    """Per-module-group L2 norm and cosine similarity vs reference direction.

    If reference is None, cosine_similarity is 1.0 (self) — useful for
    reporting bare L2 norms.
    """
    groups = ("backbone", "neck", "detect_cls", "detect_reg", "detect_dfl", "detect_other")
    results: list[GroupDrift] = []

    for g in groups:
        flat = _flatten_group(deltas, g)
        if flat.size == 0:
            continue
        l2 = float(np.linalg.norm(flat))

        if reference is not None:
            ref_flat = _flatten_group(reference, g)
            if ref_flat.size == flat.size and l2 > 0 and np.linalg.norm(ref_flat) > 0:
                cos = float(np.dot(flat, ref_flat) / (l2 * np.linalg.norm(ref_flat)))
            else:
                cos = float("nan")
        else:
            cos = 1.0

        num_params = sum(1 for name, _ in deltas if classify_parameter(name) == g)
        results.append(GroupDrift(
            module_group=g,
            l2_norm=l2,
            cosine_similarity=cos,
            num_params=num_params,
            total_numel=int(flat.size),
        ))
    return results


def summarize_drift(drifts: list[GroupDrift]) -> dict[str, dict[str, float]]:
    return {
        d.module_group: {
            "l2_norm": d.l2_norm,
            "cosine_similarity": d.cosine_similarity,
            "num_params": d.num_params,
            "total_numel": d.total_numel,
        }
        for d in drifts
    }
