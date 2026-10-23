"""Per-run results of the YOLO11s COCO-head experiments, read from the experiment registry.

The paper figures are drawn from ``research/experiment_registry/registry.yaml`` only, so a figure can never
disagree with the registered numbers. Metric per run = the pre-registered rule (mean of rounds 10 and 20,
ADR-019/021): ``results.rule_mean_r10_20`` when recorded, else the mean of ``results.round_10`` and
``results.round_20``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import yaml

SCENARIOS = ("pooled", "s5", "bus1")              # clients without bus: 0/5, 3/5, 4/5
# ADR-023 / ADR-024 runs change the data or the image size, so they get their own scenario keys and are never
# averaged with the 640 px S1b5 runs: s5moto = S1b5 rule with every motorcycle image; "_960" = 960 px;
# s9bus8 = 9 clients, bus absent on 8.
MISSING_BUS_CLIENTS = {"pooled": 0, "s5": 3, "bus1": 4, "s5moto": 3, "s5_960": 3, "s5moto_960": 3,
                       "s9bus8_960": 8}


@dataclass(frozen=True)
class Run:
    exp_id: str
    method: str
    scenario: str
    seed: int
    metrics: dict[str, float]


def _scenario(exp_id: str, text: str) -> str | None:
    s = f"{exp_id} {text}".lower()
    if "yolo11s" not in s and "y11s" not in s:
        return None
    if "s9-bus8" in s:
        base = "s9bus8"
    elif "allmoto" in s:
        base = "s5moto"
    elif "bus1" in s:
        base = "bus1"
    elif "pooled" in s:
        base = "pooled"
    elif "s1b5-3k" in s:
        base = "s5"
    else:
        return None
    key = f"{base}_960" if "960 px" in s else base
    return key if key in MISSING_BUS_CLIENTS else None


def _method(raw: str) -> str:
    return re.split(r"[\s:(]", str(raw).strip(), maxsplit=1)[0]


def _metrics(results: dict) -> dict[str, float] | None:
    """Rule metrics, plus ``results.operating_r10_20`` (TP/FP/FN, precision, recall at conf 0.25) when present."""
    r10, r20 = results.get("round_10"), results.get("round_20")
    if isinstance(results.get("rule_mean_r10_20"), dict):
        metrics = {k: float(v) for k, v in results["rule_mean_r10_20"].items()}
    elif isinstance(r10, dict) and isinstance(r20, dict):
        metrics = {k: (float(r10[k]) + float(r20[k])) / 2 for k in r10 if k in r20}
    else:
        return None
    operating = results.get("operating_r10_20")
    if isinstance(operating, dict):
        metrics = {**{k: float(v) for k, v in operating.items()}, **metrics}
    return metrics


def load_runs(registry: Path) -> list[Run]:
    """Completed YOLO11s runs on the S1b5 image set that carry rule metrics."""
    data = yaml.safe_load(Path(registry).read_text(encoding="utf-8"))
    runs = []
    for e in data.get("experiments", []):
        if e.get("status") != "completed" or e.get("seed") is None:
            continue
        scenario = _scenario(str(e.get("id", "")), str(e.get("scenario", "")))
        metrics = _metrics(e.get("results") or {})
        if scenario is None or metrics is None:
            continue
        runs.append(Run(str(e["id"]), _method(e.get("method", "")), scenario, int(e["seed"]), metrics))
    return runs


def select(runs: list[Run], method: str | None = None, scenario: str | None = None) -> list[Run]:
    return sorted((r for r in runs if (method is None or r.method == method)
                   and (scenario is None or r.scenario == scenario)), key=lambda r: r.seed)


def by_seed(runs: list[Run], method: str, scenario: str, key: str) -> dict[int, float]:
    return {r.seed: r.metrics[key] for r in select(runs, method, scenario) if key in r.metrics}


def gap_by_seed(runs: list[Run], method: str, scenario: str, key: str = "AP50_bus") -> dict[int, float]:
    """Missing-class gap = metric on the pooled re-split − metric on ``scenario``, same method and seed."""
    pooled, other = by_seed(runs, method, "pooled", key), by_seed(runs, method, scenario, key)
    return {s: pooled[s] - other[s] for s in sorted(pooled.keys() & other.keys())}


def summary(values: dict[int, float]) -> tuple[float, float, float] | None:
    """(mean, min, max) over seeds, or None when empty."""
    if not values:
        return None
    v = list(values.values())
    return mean(v), min(v), max(v)
