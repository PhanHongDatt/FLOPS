"""F3 — matched missing-class local training (CLAUDE.md §8, plan.md §6.2).

From ONE global checkpoint, train locally on a matched pair that differs only
in the target class:

* ``missing``: a client with zero target boxes (e.g. S1b C0, no bus)
* ``control``: the same client in S1-Control-Matched (same image count and
  non-target box vector, target class present; see match_report.yaml)

and record, per seed, Δθ by module / class-head row (parameter evidence) and
per-class metrics before vs after (prediction evidence). CLAUDE.md §8 requires
both halves; this module records observations only. The pass/fail rule is
pre-registered in research/feasibility/F3/README.md and is applied to the
saved artifacts, not here.

Training and evaluation are injected through :class:`F3Hooks` so the pipeline
is testable without Ultralytics; ``scripts/run_f3.py`` supplies the real ones.
"""
from __future__ import annotations

import csv
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.partitioner import load_partition_manifest
from src.evaluation.drift import matched_pair_update_analysis
from src.utils.artifacts import save_yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)

SIDES = ("missing", "control")


@dataclass(frozen=True)
class F3Side:
    role: str                # "missing" | "control"
    partition_id: str
    client_id: str
    data_yaml: Path
    target_boxes: int


@dataclass(frozen=True)
class F3Hooks:
    build: Callable[[], Any]                                  # fresh model from the global checkpoint
    train: Callable[[Any, Path, int, Path, str], None]        # (model, data_yaml, seed, project, name)
    evaluate: Callable[[Any], dict[str, float]]               # on the shared global val set
    state: Callable[[Any], dict[str, np.ndarray]]             # ordered state_dict as numpy


def resolve_side(role: str, manifest_path: Path, client_id: str, target_class: str) -> F3Side:
    """Read one side of the pair from a partition manifest written by generate_partition.py."""
    manifest = load_partition_manifest(Path(manifest_path))
    if client_id not in manifest.class_counts:
        raise KeyError(f"client {client_id} not in {manifest_path} ({sorted(manifest.class_counts)})")
    data_yaml = Path(manifest_path).parent / f"data_{client_id}.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} missing — run scripts/generate_partition.py first")
    return F3Side(
        role=role,
        partition_id=manifest.partition_id,
        client_id=client_id,
        data_yaml=data_yaml,
        target_boxes=int(manifest.class_counts[client_id].get(target_class, 0)),
    )


def validate_pair(missing: F3Side, control: F3Side) -> None:
    if missing.target_boxes != 0:
        raise ValueError(
            f"missing side {missing.partition_id}/{missing.client_id} must have zero target "
            f"boxes, has {missing.target_boxes}"
        )
    if control.target_boxes <= 0:
        raise ValueError(
            f"control side {control.partition_id}/{control.client_id} must have positive "
            "target boxes, has 0"
        )


def _states_equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def _mean_std(values: list[float]) -> dict[str, float]:
    finite = [v for v in values if not math.isnan(v)]
    n = len(finite)
    return {
        "mean": float(np.mean(finite)) if n else float("nan"),
        # ddof=1: across-seed sample std, the quantity the pre-registered rule uses
        "std": float(np.std(finite, ddof=1)) if n > 1 else float("nan"),
        "n": n,
    }


def _train_side(
    side: F3Side,
    seed: int,
    hooks: F3Hooks,
    global_state: dict[str, np.ndarray],
    out_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    model = hooks.build()
    if not _states_equal(hooks.state(model), global_state):
        raise RuntimeError(
            "build() did not return the same global model for every side/seed; "
            "Δθ would mix initialisation noise into the update"
        )
    hooks.train(model, side.data_yaml, seed, out_dir / "train", f"{side.role}_seed{seed}")
    return hooks.state(model), hooks.evaluate(model)


def _metric_rows(seed: int, side: F3Side, before: dict[str, float], after: dict[str, float]) -> list[dict]:
    return [
        {
            "seed": seed, "side": side.role, "partition_id": side.partition_id,
            "client_id": side.client_id, "metric": k,
            "before": before[k], "after": after[k], "delta": after[k] - before[k],
        }
        for k in sorted(before.keys() & after.keys())
    ]


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(
    analyses: dict[int, dict[str, Any]],
    rows: list[dict],
    before: dict[str, float],
    target_class: str,
) -> dict[str, Any]:
    parameter = {
        key: _mean_std([a["target_contrast"][key] for a in analyses.values()])
        for key in next(iter(analyses.values()))["target_contrast"]
        if key != "target_class"
    }
    prediction: dict[str, Any] = {}
    for metric in sorted({r["metric"] for r in rows}):
        per_side = {
            s: {r["seed"]: r["delta"] for r in rows if r["metric"] == metric and r["side"] == s}
            for s in SIDES
        }
        seeds = sorted(per_side["missing"].keys() & per_side["control"].keys())
        prediction[metric] = {
            "delta_missing": _mean_std([per_side["missing"][s] for s in seeds]),
            "delta_control": _mean_std([per_side["control"][s] for s in seeds]),
            "delta_gap": _mean_std([per_side["missing"][s] - per_side["control"][s] for s in seeds]),
        }
    known = any(before.get(k, 0.0) > 0 for k in (f"AP50_{target_class}", f"AP_{target_class}"))
    return {"parameter": parameter, "prediction": prediction, "target_known_before": known}


def run_matched_pair(
    missing: F3Side,
    control: F3Side,
    target_class: str,
    class_names: Sequence[str],
    seeds: Sequence[int],
    hooks: F3Hooks,
    out_dir: Path,
) -> dict[str, Any]:
    """Run F3 for every seed and write metrics.csv, update_analysis_seed*.yaml, summary.yaml."""
    validate_pair(missing, control)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_model = hooks.build()
    global_state = hooks.state(global_model)
    # Evaluation of a fixed model is deterministic, so "before" is measured once.
    before = hooks.evaluate(global_model)
    del global_model

    rows: list[dict] = []
    analyses: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        trained: dict[str, dict[str, np.ndarray]] = {}
        for side in (missing, control):
            trained[side.role], after = _train_side(side, seed, hooks, global_state, out_dir)
            rows.extend(_metric_rows(seed, side, before, after))
        analyses[seed] = matched_pair_update_analysis(
            global_state, trained["missing"], trained["control"], target_class, class_names
        )
        save_yaml(analyses[seed], out_dir / f"update_analysis_seed{seed}.yaml")
        logger.info("F3 seed %d target contrast: %s", seed, analyses[seed]["target_contrast"])

    if not rows:
        raise ValueError(
            "evaluate() returned no metric present both before and after training; "
            "nothing to compare (check the eval data yaml and class names)"
        )
    _write_csv(rows, out_dir / "metrics.csv")
    summary = {
        "target_class": target_class,
        "seeds": list(seeds),
        "missing": {"partition_id": missing.partition_id, "client_id": missing.client_id},
        "control": {"partition_id": control.partition_id, "client_id": control.client_id},
        "before": before,
        **_summarize(analyses, rows, before, target_class),
    }
    if not summary["target_known_before"]:
        logger.warning(
            "Global checkpoint has zero AP for %r before local training: forgetting is not "
            "measurable from this initialisation. Do not read a null result as evidence "
            "against H1 (plan.md §6.2).", target_class,
        )
    save_yaml(summary, out_dir / "summary.yaml")
    return summary
