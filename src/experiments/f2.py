"""F2 — controlled perturbation of class-associated parameters (CLAUDE.md §8 F2).

Perturb ONLY a candidate parameter group of a trained nc=4 detector and
measure per-class effects:

* ``row_noise``      — Gaussian noise on class row j of the 3 class-head WEIGHT
                       keys, std = ``rel`` x RMS of that row, per key   [ENGINEERING].
                       Biases are excluded: a bias row is one scalar (~-8.5 after
                       bias_init), so relative noise would move the logit by
                       several units and swamp the weight effect.
* ``row_bias_shift`` — subtract ``delta`` from class j's bias at every scale.
                       Positive control: it lowers only class j's logits, so it
                       checks that the evaluation sees class-specific effects.
* ``shared_noise``   — norm-matched control: Gaussian noise on the SHARED
                       cv3.<s>.1 convs with the same total L2 as the target row's
                       ``row_noise`` at that level and seed.

Perturbing every row j and measuring every class i gives an effect matrix
E[j][i] = ΔAP50_i; class specificity shows as a dominant diagonal. This module
records observations only; the decision rule is pre-registered in
research/feasibility/F2/README.md.
"""
from __future__ import annotations

import csv
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.model.parameter_map import class_head_keys
from src.utils.artifacts import save_yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Shared conv right before the class head, in every scale: cv3.<s>.1 (Conv+BN).
_SHARED_CONTROL_RE = re.compile(r"^model\.\d+\.cv3\.\d+\.1\.conv\.weight$")

State = dict[str, np.ndarray]


@dataclass(frozen=True)
class F2Spec:
    kind: str               # row_noise | row_bias_shift | shared_noise
    row: str | None         # perturbed class row; None for shared_noise
    level: float            # rel (noise) or delta (bias shift)
    seed: int | None        # None for the deterministic bias shift
    match_row: str | None = None   # shared_noise: row whose noise L2 it matches

    @property
    def spec_id(self) -> str:
        parts = [self.kind, self.row or f"match-{self.match_row}", str(self.level)]
        if self.seed is not None:
            parts.append(f"s{self.seed}")
        return "_".join(parts)


@dataclass(frozen=True)
class F2Hooks:
    build: Callable[[], Any]                       # trained checkpoint under test
    state: Callable[[Any], State]
    load: Callable[[Any, State], None]
    evaluate: Callable[[Any], dict[str, float]]    # fixed val subset
    confidence: Callable[[Any], dict[str, float]]  # per-class prediction confidence


def _copy(state: State) -> State:
    return {k: np.array(v, copy=True) for k, v in state.items()}


def _head(state: State, nc: int) -> list[str]:
    return class_head_keys(list(state), [np.shape(v) for v in state.values()], nc)


def class_row_noise(state: State, nc: int, cls_idx: int, rel: float, seed: int) -> tuple[State, float]:
    """Return (perturbed copy, L2 of the added noise)."""
    rng = np.random.default_rng(seed)
    out = _copy(state)
    sq = 0.0
    for k in (k for k in _head(state, nc) if k.endswith(".weight")):
        row = np.asarray(state[k][cls_idx], dtype=np.float64)
        rms = float(np.sqrt(np.mean(row ** 2)))
        noise = rng.standard_normal(row.shape) * rel * rms
        out[k][cls_idx] = row + noise
        sq += float(np.sum(noise ** 2))
    return out, math.sqrt(sq)


def class_row_bias_shift(state: State, nc: int, cls_idx: int, delta: float) -> tuple[State, float]:
    out = _copy(state)
    keys = [k for k in _head(state, nc) if k.endswith(".bias")]
    for k in keys:
        out[k][cls_idx] = state[k][cls_idx] - delta
    return out, abs(delta) * math.sqrt(len(keys))


def shared_noise(state: State, target_l2: float, seed: int) -> tuple[State, float]:
    keys = [k for k in state if _SHARED_CONTROL_RE.match(k)]
    if not keys:
        raise ValueError(f"no shared control params match {_SHARED_CONTROL_RE.pattern!r}")
    rng = np.random.default_rng(seed)
    draws = {k: rng.standard_normal(np.shape(state[k])) for k in keys}
    norm = math.sqrt(sum(float(np.sum(d ** 2)) for d in draws.values()))
    out = _copy(state)
    for k in keys:
        out[k] = np.asarray(state[k], dtype=np.float64) + draws[k] * (target_l2 / norm)
    return out, float(target_l2)


def build_specs(cfg: dict[str, Any], class_names: Sequence[str], target_class: str) -> list[F2Spec]:
    if target_class not in class_names:
        raise ValueError(f"target_class {target_class!r} not in {list(class_names)}")
    seeds = list(cfg.get("seeds", []))
    specs = [
        F2Spec("row_noise", row, float(rel), int(seed))
        for rel in cfg.get("noise_rel", []) for row in class_names for seed in seeds
    ]
    specs += [F2Spec("row_bias_shift", row, float(d), None)
              for d in cfg.get("bias_shift", []) for row in class_names]
    if cfg.get("shared_control", False):
        specs += [F2Spec("shared_noise", None, float(rel), int(seed), match_row=target_class)
                  for rel in cfg.get("noise_rel", []) for seed in seeds]
    return specs


def _apply(spec: F2Spec, base: State, class_names: Sequence[str]) -> tuple[State, float]:
    nc = len(class_names)
    if spec.kind == "row_bias_shift" and spec.row is not None:
        return class_row_bias_shift(base, nc, class_names.index(spec.row), spec.level)
    if spec.seed is None:
        raise ValueError(f"{spec.spec_id}: noise perturbations need a seed")
    if spec.kind == "row_noise" and spec.row is not None:
        return class_row_noise(base, nc, class_names.index(spec.row), spec.level, spec.seed)
    if spec.kind == "shared_noise" and spec.match_row is not None:
        _, l2 = class_row_noise(base, nc, class_names.index(spec.match_row), spec.level, spec.seed)
        return shared_noise(base, l2, spec.seed)
    raise ValueError(f"invalid perturbation spec {spec!r}")


def _mean_std(values: list[float]) -> dict[str, float]:
    n = len(values)
    return {"mean": float(np.mean(values)) if n else float("nan"),
            "std": float(np.std(values, ddof=1)) if n > 1 else float("nan"), "n": n}


def _effect_matrix(rows: list[dict], class_names: Sequence[str], metric: str) -> dict[str, Any]:
    """matrix[kind][level][perturbed row][class] = mean/std over seeds of Δmetric."""
    out: dict[str, Any] = {}
    for r in rows:
        if not r["metric"].startswith(f"{metric}_"):
            continue
        cls = r["metric"][len(metric) + 1:]
        if cls not in class_names:
            continue
        key = r["row"] or "shared"
        (out.setdefault(r["kind"], {}).setdefault(str(r["level"]), {})
            .setdefault(key, {}).setdefault(cls, []).append(r["delta"]))
    return {kind: {lvl: {row: {c: _mean_std(v) for c, v in cls.items()} for row, cls in rows_.items()}
                   for lvl, rows_ in lvls.items()} for kind, lvls in out.items()}


def _measure(model: Any, hooks: F2Hooks) -> dict[str, float]:
    # plain floats: numpy scalars would be written as python/object tags in YAML
    return {k: float(v) for k, v in {**hooks.evaluate(model), **hooks.confidence(model)}.items()}


def run_perturbations(
    specs: Sequence[F2Spec],
    class_names: Sequence[str],
    hooks: F2Hooks,
    out_dir: Path,
    effect_metric: str = "AP50",
) -> dict[str, Any]:
    """Evaluate every spec against the same unperturbed checkpoint; write artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = hooks.build()
    base = hooks.state(model)
    baseline = _measure(model, hooks)

    rows: list[dict] = []
    log: list[dict] = []
    try:
        for spec in specs:
            perturbed, l2 = _apply(spec, base, class_names)
            hooks.load(model, perturbed)
            values = _measure(model, hooks)
            log.append({**asdict(spec), "spec_id": spec.spec_id, "l2": l2})
            rows.extend(
                {"spec_id": spec.spec_id, "kind": spec.kind, "row": spec.row or "",
                 "level": spec.level, "seed": "" if spec.seed is None else spec.seed,
                 "metric": k, "base": baseline[k], "value": values[k],
                 "delta": values[k] - baseline[k]}
                for k in sorted(baseline.keys() & values.keys())
            )
            logger.info("F2 %s (L2=%.4g) done", spec.spec_id, l2)
    finally:
        hooks.load(model, base)

    if rows:
        with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save_yaml(log, out_dir / "perturbation_log.yaml")
    summary = {
        "effect_metric": effect_metric,
        "baseline": baseline,
        "effect_matrix": _effect_matrix(rows, class_names, effect_metric),
    }
    save_yaml(summary, out_dir / "summary.yaml")
    return summary
