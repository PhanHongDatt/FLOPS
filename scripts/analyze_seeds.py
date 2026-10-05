"""Multi-seed summary for the S1b-2k comparison (ADR-011, pre-declared rule).

Per run: mean of the server evals at rounds 20, 25, 30 (declared before s5a was read).
Per arm: mean ± sample std over seeds. Effects are PAIRED by seed (same partition, same
training seed): H1 = A0@S1b − A0@control; method = arm − A0.

Usage:
  python scripts/analyze_seeds.py kaggle/output/s5a kaggle/output/s5b kaggle/output/s6a kaggle/output/s6b
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

RULE_ROUNDS = (20, 25, 30)
METRICS = ("mAP50", "AP50_bus", "AP50_truck", "AP50_car", "FP_bus", "FN_bus")


def rule_value(rows: list[dict[str, Any]], metric: str) -> float | None:
    by_round = {int(float(r["round"])): r for r in rows}
    if not all(r in by_round and by_round[r].get(metric) not in (None, "") for r in RULE_ROUNDS):
        return None
    return sum(float(by_round[r][metric]) for r in RULE_ROUNDS) / len(RULE_ROUNDS)


def _arm(env: dict[str, Any]) -> str:
    arm, pid = str(env.get("exp_id")), str(env.get("partition_id"))
    for tag in ("control", "pooled"):          # same exp_id on another partition is another arm
        if tag in pid:
            return f"{arm}@{tag}"
    return arm


def summarise(roots: list[Path], metric: str) -> dict[str, dict[str, Any]]:
    per_arm: dict[str, dict[int, float]] = {}
    for root in roots:
        for d in sorted((Path(root) / "artifacts" / "runs").glob("*_2k_seed42")):
            env = json.loads((d / "environment.json").read_text(encoding="utf-8"))
            rows = list(csv.DictReader((d / "round_metrics.csv").open(encoding="utf-8")))
            v = rule_value(rows, metric)
            if v is not None:
                per_arm.setdefault(_arm(env), {})[int(env["seed"])] = v
    out = {}
    for arm, by_seed in per_arm.items():
        vals = list(by_seed.values())
        out[arm] = {"per_seed": dict(sorted(by_seed.items())), "n": len(vals), "mean": statistics.fmean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else float("nan")}
    return out


def paired_effect(table: dict[str, dict[str, Any]], arm: str, ref: str) -> dict[str, Any]:
    a, b = table[arm]["per_seed"], table[ref]["per_seed"]
    diffs = {s: a[s] - b[s] for s in sorted(a.keys() & b.keys())}
    vals = list(diffs.values())
    return {"per_seed": diffs, "n": len(vals), "mean": statistics.fmean(vals) if vals else float("nan"),
            "std": statistics.stdev(vals) if len(vals) > 1 else float("nan")}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")   # "±" / "−" on a cp1252 Windows console
    roots = [Path(p) for p in sys.argv[1:]]
    for metric in METRICS:
        table = summarise(roots, metric)
        if not table:
            continue
        print(f"\n## {metric} (mean of rounds 20/25/30; mean ± std over seeds)")
        for arm, t in sorted(table.items()):
            seeds = ", ".join(f"{s}:{v:.4f}" for s, v in t["per_seed"].items())
            print(f"{arm:20s} n={t['n']} mean={t['mean']:.4f} std={t['std']:.4f}  [{seeds}]")
        if {"A0", "A0@control"} <= table.keys():
            e = paired_effect(table, "A0", "A0@control")
            print(f"H1 (A0@S1b − A0@control): {e['mean']:+.4f} ± {e['std']:.4f} (n={e['n']}) {e['per_seed']}")
        for arm in sorted(table):
            if arm not in ("A0", "A0@control") and "A0" in table:
                e = paired_effect(table, arm, "A0")
                if e["n"]:
                    print(f"{arm} − A0: {e['mean']:+.4f} ± {e['std']:.4f} (n={e['n']}) {e['per_seed']}")


if __name__ == "__main__":
    main()
