"""Parameter drift analysis for G4 and F3.

Section 8 (F3) and Section 13 of CLAUDE.md require parameter/update evidence
BEYOND L2 norm alone. This module computes:
  - Δθ = θ_local - θ_global by module group (backbone/neck/detect_cls/detect_reg)
    plus ``class_head``: the 6 final-cv3 keys, the only class-specific subset
    of detect_cls (780 of 370,962 elements at nc=4, F1 runtime 2026-10-02)
  - per-class row drift of class_head (L2, signed mean Δweight / Δbias, cosine)
  - the F3 matched-pair contrast (client without target vs matched control)
  - L2 norm per group
  - cosine similarity per group (flattened)
  - Direction (angle proxy)

A Missing-Class claim requires parameter evidence + prediction evidence.
This module handles the parameter half only.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.model.parameter_map import (
    class_head_keys,
    classify_parameter,
    is_class_specific,
)


@dataclass(frozen=True)
class GroupDrift:
    module_group: str
    l2_norm: float
    cosine_similarity: float
    num_params: int
    total_numel: int


@dataclass(frozen=True)
class ClassRowDrift:
    class_name: str
    l2_norm: float
    weight_delta_mean: float     # signed mean over the row's weights, all scales
    bias_delta_mean: float       # signed mean over the 3 per-scale biases
    cosine_vs_reference: float   # NaN when either row delta is zero / no reference
    numel: int


_BN_STATS = (".running_mean", ".running_var")
_COUNTERS = (".num_batches_tracked",)


def _is_bn_stat(name: str) -> bool:
    return name.endswith(_BN_STATS)


def _is_learned(name: str) -> bool:
    return not name.endswith(_BN_STATS + _COUNTERS)


# Module groups hold LEARNED parameters only. BN running statistics are data
# statistics, reported as "bn_stats"; num_batches_tracked counters are dropped.
# "class_head" overlaps detect_cls on purpose: module groups keep their meaning
# across reports, and the class-specific subset is reported next to them.
def _module_member(group: str) -> Callable[[str], bool]:
    def member(name: str) -> bool:
        return _is_learned(name) and classify_parameter(name) == group
    return member


_GROUPS: dict[str, Callable[[str], bool]] = {
    **{
        g: _module_member(g)
        for g in ("backbone", "neck", "detect_cls", "detect_reg", "detect_dfl", "detect_other")
    },
    "class_head": is_class_specific,
    "bn_stats": _is_bn_stat,
}


def _flatten_group(
    named_deltas: list[tuple[str, np.ndarray]],
    group: str,
) -> np.ndarray:
    member = _GROUPS[group]
    parts = [np.asarray(d, dtype=np.float64).ravel() for name, d in named_deltas if member(name)]
    if not parts:
        return np.zeros(0)
    return np.concatenate(parts)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if a.size != b.size or na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


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
    results: list[GroupDrift] = []

    for g, member in _GROUPS.items():
        flat = _flatten_group(deltas, g)
        if flat.size == 0:
            continue
        l2 = float(np.linalg.norm(flat))
        cos = _cosine(flat, _flatten_group(reference, g)) if reference is not None else 1.0
        num_params = sum(1 for name, _ in deltas if member(name))
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


@dataclass(frozen=True)
class _ClassSlices:
    weights: list[np.ndarray]   # per class, weight-row elements over all scales
    biases: list[np.ndarray]    # per class, the per-scale bias elements


def _class_slices(deltas: list[tuple[str, np.ndarray]], nc: int) -> _ClassSlices:
    by_name = dict(deltas)
    keys = class_head_keys(list(by_name), [np.shape(by_name[k]) for k in by_name], nc)
    weights: list[list[np.ndarray]] = [[] for _ in range(nc)]
    biases: list[list[np.ndarray]] = [[] for _ in range(nc)]
    for k in keys:
        arr = np.asarray(by_name[k], dtype=np.float64)
        target = biases if k.endswith(".bias") else weights
        for c in range(nc):
            target[c].append(np.ravel(arr[c]))
    return _ClassSlices(
        weights=[np.concatenate(w) for w in weights],
        biases=[np.concatenate(b) for b in biases],
    )


def class_row_drift(
    deltas: list[tuple[str, np.ndarray]],
    class_names: Sequence[str],
    reference: list[tuple[str, np.ndarray]] | None = None,
) -> list[ClassRowDrift]:
    """Drift of each class's slice (dim 0) of the 6 class-head keys.

    Raises ValueError unless exactly 6 class-head keys with dim0 == nc exist,
    so a wrong model (nc=80, shifted Detect index) cannot yield empty rows.
    """
    nc = len(class_names)
    cur = _class_slices(deltas, nc)
    ref = _class_slices(reference, nc) if reference is not None else None
    out: list[ClassRowDrift] = []
    for c, name in enumerate(class_names):
        row = np.concatenate([cur.weights[c], cur.biases[c]])
        ref_row = np.concatenate([ref.weights[c], ref.biases[c]]) if ref is not None else None
        out.append(ClassRowDrift(
            class_name=name,
            l2_norm=float(np.linalg.norm(row)),
            weight_delta_mean=float(cur.weights[c].mean()),
            bias_delta_mean=float(cur.biases[c].mean()),
            cosine_vs_reference=_cosine(row, ref_row) if ref_row is not None else float("nan"),
            numel=int(row.size),
        ))
    return out


def _rows_as_dict(rows: list[ClassRowDrift]) -> dict[str, dict[str, float]]:
    return {
        r.class_name: {
            "l2_norm": r.l2_norm,
            "weight_delta_mean": r.weight_delta_mean,
            "bias_delta_mean": r.bias_delta_mean,
            "cosine_vs_reference": r.cosine_vs_reference,
            "numel": r.numel,
        }
        for r in rows
    }


def _target_row_l2_ratio(rows: list[ClassRowDrift], target: str) -> float:
    """L2 of the target row over the mean L2 of the non-target rows."""
    t = next(r.l2_norm for r in rows if r.class_name == target)
    others = [r.l2_norm for r in rows if r.class_name != target]
    mean_other = float(np.mean(others)) if others else 0.0
    return t / mean_other if mean_other > 0 else float("nan")


def matched_pair_update_analysis(
    global_state: dict[str, np.ndarray],
    missing_state: dict[str, np.ndarray],
    control_state: dict[str, np.ndarray],
    target_class: str,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """F3 parameter evidence for one seed: Δθ of the client WITHOUT the target
    class vs Δθ of its matched control, both trained from ``global_state``.

    Reports observations only. The pass/fail rule is pre-registered in
    research/feasibility/F3/README.md and applied across seeds, never here.
    Prediction-level evidence (ΔAP, confidence, FP/FN) is collected by the F3
    script; CLAUDE.md §8 requires both halves.
    """
    if target_class not in class_names:
        raise ValueError(f"target_class {target_class!r} not in {list(class_names)}")

    d_missing = compute_delta(missing_state, global_state)
    d_control = compute_delta(control_state, global_state)
    rows_m = class_row_drift(d_missing, class_names, reference=d_control)
    rows_c = class_row_drift(d_control, class_names)
    t = class_names.index(target_class)

    bias_m = rows_m[t].bias_delta_mean
    bias_c = rows_c[t].bias_delta_mean
    return {
        "module_drift": {
            "missing": summarize_drift(group_drift(d_missing)),
            "control": summarize_drift(group_drift(d_control)),
            "cosine_missing_vs_control": {
                d.module_group: d.cosine_similarity
                for d in group_drift(d_missing, reference=d_control)
            },
        },
        "class_rows": {
            "missing": _rows_as_dict(rows_m),
            "control": _rows_as_dict(rows_c),
        },
        "target_contrast": {
            "target_class": target_class,
            "bias_delta_mean_missing": bias_m,
            "bias_delta_mean_control": bias_c,
            "bias_delta_gap": bias_m - bias_c,
            "target_row_l2_missing": rows_m[t].l2_norm,
            "target_row_l2_control": rows_c[t].l2_norm,
            "target_row_l2_ratio_missing": _target_row_l2_ratio(rows_m, target_class),
            "target_row_l2_ratio_control": _target_row_l2_ratio(rows_c, target_class),
            "target_row_cosine_missing_vs_control": rows_m[t].cosine_vs_reference,
        },
    }
