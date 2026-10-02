"""Cross-run comparison: aggregate arm × seed runs into reportable tables.

Why this exists: each invocation of `scripts/run_fl_experiment.py` produces one
arm's run directory and knows nothing about the others. Comparing methods is a
separate step, and CLAUDE.md §14/§20 put hard conditions on it:

* every compared arm must use the **same partition**, the same seed set, the same
  number of rounds and the same evaluator settings;
* the **final round** is reported, not the best round;
* no seed may be dropped silently.

Those conditions are enforced here in code (`check_comparable`) rather than left
to discipline, because "compare methods trained on different partitions and call
the difference algorithmic" is exactly the failure §20 names.

Pure stdlib + numpy: runnable and testable with no GPU, torch or ultralytics.
"""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.data.bdd100k import TARGET_CLASSES

# Config fields that MUST be identical across arms for a comparison to be valid.
# fraction_evaluate is deliberately absent: it only controls client-side
# evaluation, which is not the reporting path, so it cannot bias the comparison.
PROTOCOL_KEYS: tuple[str, ...] = (
    "num_rounds",
    "local_epochs",
    "num_clients",
    "fraction_fit",
    "batch_size",
    "image_size",
    "lr0",
    "conf",
    "iou",
    "weights",
)

PRIMARY_METRICS: tuple[str, ...] = ("mAP50", "mAP50-95")


@dataclass(frozen=True)
class RunRecord:
    """One completed run directory, reduced to what a comparison needs."""

    run_dir: Path
    arm: str                       # exp_id, e.g. "A0" or "A4b_rho0.25"
    seed: int
    partition_id: str
    run_class: str                 # smoke | feasibility | main
    protocol: dict[str, Any]       # PROTOCOL_KEYS subset from config.yaml
    final_metrics: dict[str, float]
    rounds: list[dict[str, float]] = field(default_factory=list)
    git_commit: str = "unknown"

    @property
    def final_round(self) -> int | None:
        if not self.rounds:
            return None
        return int(max(r["round"] for r in self.rounds))


def _read_metrics_csv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name, val = row.get("metric"), row.get("value")
            if not name or val in (None, "", "None"):
                continue
            try:
                out[name] = float(val)
            except ValueError:
                continue
    return out


def _read_round_metrics(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    with path.open("r", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: dict[str, float] = {}
            for k, v in raw.items():
                if k is None or v in (None, "", "None"):
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    continue
            if "round" in row:
                rows.append(row)
    return sorted(rows, key=lambda r: r["round"])


def _extract_protocol(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {}) or {}
    train = config.get("train", {}) or {}
    fed = config.get("federated", {}) or {}
    ev = config.get("evaluation", {}) or {}
    return {
        "num_rounds": fed.get("num_rounds"),
        "local_epochs": fed.get("local_epochs"),
        "num_clients": fed.get("num_clients"),
        "fraction_fit": fed.get("fraction_fit"),
        "batch_size": train.get("batch_size"),
        "image_size": model.get("image_size"),
        "lr0": train.get("lr0"),
        "conf": ev.get("conf", train.get("conf")),
        "iou": ev.get("iou", train.get("iou")),
        "weights": model.get("weights"),
    }


def load_run(run_dir: Path) -> RunRecord:
    """Read one run directory into a RunRecord.

    Raises:
        FileNotFoundError: when the directory is missing environment.json or
            config.yaml, i.e. it is not a run produced by `prepare_run`.
    """
    env_path = run_dir / "environment.json"
    cfg_path = run_dir / "config.yaml"
    if not env_path.exists() or not cfg_path.exists():
        raise FileNotFoundError(
            f"{run_dir} is not a run directory (missing environment.json/config.yaml)"
        )
    env = json.loads(env_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    return RunRecord(
        run_dir=run_dir,
        arm=str(env.get("exp_id", run_dir.name)),
        seed=int(env.get("seed", -1)),
        partition_id=str(env.get("partition_id", "")),
        run_class=str(env.get("run_class", "")),
        protocol=_extract_protocol(config),
        final_metrics=_read_metrics_csv(run_dir / "metrics.csv"),
        rounds=_read_round_metrics(run_dir / "round_metrics.csv"),
        git_commit=str(env.get("git_commit", "unknown")),
    )


def discover_runs(runs_root: Path) -> list[RunRecord]:
    """Load every run directory under ``runs_root`` that looks like a run."""
    records: list[RunRecord] = []
    if not runs_root.is_dir():
        return records
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            records.append(load_run(d))
        except FileNotFoundError:
            continue
    return records


# ── validity guards (CLAUDE.md §14, §20) ──────────────────────────────────
def check_comparable(records: Iterable[RunRecord]) -> list[str]:
    """Return human-readable reasons the given runs must NOT be compared.

    Empty list means the comparison satisfies §14/§20 on the mechanical points:
    one partition, one protocol, one seed set per arm, and no arm reporting a
    different number of rounds.
    """
    recs = list(records)
    problems: list[str] = []
    if not recs:
        return ["no runs given"]

    partitions = sorted({r.partition_id for r in recs})
    if len(partitions) > 1:
        problems.append(
            "runs span multiple partitions "
            f"{partitions} — a difference between them is not algorithmic (§20)"
        )

    run_classes = sorted({r.run_class for r in recs})
    if len(run_classes) > 1:
        problems.append(f"runs mix run classes {run_classes}")
    if run_classes and run_classes[0] == "smoke":
        problems.append("run_class=smoke results are correctness-only, not reportable")

    for key in PROTOCOL_KEYS:
        values = {json.dumps(r.protocol.get(key), sort_keys=True) for r in recs}
        if len(values) > 1:
            problems.append(f"protocol mismatch on {key!r}: {sorted(values)}")

    by_arm: dict[str, set[int]] = {}
    for r in recs:
        by_arm.setdefault(r.arm, set()).add(r.seed)
    seed_sets = {arm: tuple(sorted(s)) for arm, s in by_arm.items()}
    if len(set(seed_sets.values())) > 1:
        problems.append(
            "arms do not share the same seed set: "
            + ", ".join(f"{a}={list(s)}" for a, s in sorted(seed_sets.items()))
            + " — dropping a seed for one arm only is cherry-picking (§20)"
        )

    final_rounds = {r.arm: {r2.final_round for r2 in recs if r2.arm == r.arm} for r in recs}
    for arm, rounds in sorted(final_rounds.items()):
        if len(rounds) > 1:
            problems.append(f"arm {arm} has runs ending at different rounds {sorted(rounds)}")
        elif rounds and next(iter(rounds)) is None:
            problems.append(f"arm {arm} has no round_metrics.csv — run incomplete")

    return problems


def min_seeds_warning(records: Iterable[RunRecord], min_seeds: int = 3) -> list[str]:
    """Arms with fewer than ``min_seeds`` seeds (CLAUDE.md §14 requires 3 for main)."""
    by_arm: dict[str, set[int]] = {}
    for r in records:
        by_arm.setdefault(r.arm, set()).add(r.seed)
    return [
        f"{arm} has {len(seeds)} seed(s) {sorted(seeds)} < {min_seeds}"
        for arm, seeds in sorted(by_arm.items())
        if len(seeds) < min_seeds
    ]


# ── aggregation ───────────────────────────────────────────────────────────
def metric_keys(records: Iterable[RunRecord]) -> list[str]:
    """Reported metric names: primary mAP plus per-class columns, in a fixed order."""
    present: set[str] = set()
    for r in records:
        present.update(r.final_metrics)
    ordered: list[str] = [m for m in PRIMARY_METRICS if m in present]
    for prefix in ("AP50", "AP", "precision", "recall", "TP", "FP", "FN"):
        for cls in TARGET_CLASSES:
            key = f"{prefix}_{cls}"
            if key in present:
                ordered.append(key)
    return ordered


def aggregate_by_arm(
    records: Iterable[RunRecord],
) -> dict[str, dict[str, tuple[float, float, int]]]:
    """``{arm: {metric: (mean, std, n_seeds)}}`` using the FINAL round of each run.

    ``std`` is the sample standard deviation and is 0.0 for a single seed. Reported
    as ``mean ± std`` per §14; the final round is used because reporting the best
    round per method without declaring the rule is forbidden (§20).
    """
    recs = list(records)
    keys = metric_keys(recs)
    out: dict[str, dict[str, tuple[float, float, int]]] = {}
    for r in recs:
        out.setdefault(r.arm, {})
    for arm in out:
        arm_recs = [r for r in recs if r.arm == arm]
        for key in keys:
            vals = [r.final_metrics[key] for r in arm_recs if key in r.final_metrics]
            if not vals:
                continue
            mean = statistics.fmean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            out[arm][key] = (mean, std, len(vals))
    return out


def delta_vs_baseline(
    aggregated: dict[str, dict[str, tuple[float, float, int]]],
    baseline_arm: str,
) -> dict[str, dict[str, float]]:
    """``{arm: {metric: mean(arm) - mean(baseline)}}`` — the ΔAP_c of §13.

    Raises:
        KeyError: when the baseline arm is absent. A comparison without its
            declared baseline is not a comparison.
    """
    if baseline_arm not in aggregated:
        raise KeyError(
            f"baseline arm {baseline_arm!r} not among runs {sorted(aggregated)}"
        )
    base = aggregated[baseline_arm]
    return {
        arm: {
            key: vals[0] - base[key][0]
            for key, vals in metrics.items()
            if key in base
        }
        for arm, metrics in aggregated.items()
        if arm != baseline_arm
    }


def scenario_delta(
    records: Iterable[RunRecord],
    arm: str,
    partition_a: str,
    partition_b: str,
) -> dict[str, tuple[float, float, int]]:
    """Paired per-seed delta of one arm across two partitions (the C1 comparison).

    Used for S1 vs S1-Control, where the same arm and seed is run on two
    partitions and the difference is attributed to the partition. Pairing by seed
    is what makes the std meaningful; seeds present on only one side are skipped
    and reflected in the returned count.
    """
    recs = list(records)
    a = {r.seed: r for r in recs if r.arm == arm and r.partition_id == partition_a}
    b = {r.seed: r for r in recs if r.arm == arm and r.partition_id == partition_b}
    shared = sorted(set(a) & set(b))
    keys = metric_keys(recs)
    out: dict[str, tuple[float, float, int]] = {}
    for key in keys:
        diffs = [
            a[s].final_metrics[key] - b[s].final_metrics[key]
            for s in shared
            if key in a[s].final_metrics and key in b[s].final_metrics
        ]
        if not diffs:
            continue
        mean = statistics.fmean(diffs)
        std = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        out[key] = (mean, std, len(diffs))
    return out


# ── writers ───────────────────────────────────────────────────────────────
def write_per_run_csv(records: Iterable[RunRecord], path: Path) -> None:
    recs = list(records)
    keys = metric_keys(recs)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["arm", "seed", "partition_id", "run_class", "final_round",
                  "git_commit", "run_dir"] + keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(recs, key=lambda x: (x.arm, x.seed)):
            row: dict[str, Any] = {
                "arm": r.arm, "seed": r.seed, "partition_id": r.partition_id,
                "run_class": r.run_class, "final_round": r.final_round,
                "git_commit": r.git_commit, "run_dir": r.run_dir.name,
            }
            row.update({k: r.final_metrics.get(k, "") for k in keys})
            w.writerow(row)


def write_summary_csv(
    aggregated: dict[str, dict[str, tuple[float, float, int]]],
    path: Path,
    deltas: dict[str, dict[str, float]] | None = None,
    baseline_arm: str | None = None,
) -> None:
    """One row per (arm, metric) with mean, std, n and optional delta vs baseline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["arm", "metric", "mean", "std", "n_seeds",
                        "mean_pm_std", "delta_vs_baseline", "baseline_arm"],
        )
        w.writeheader()
        for arm in sorted(aggregated):
            for metric, (mean, std, n) in aggregated[arm].items():
                delta = ""
                if deltas and arm in deltas and metric in deltas[arm]:
                    delta = f"{deltas[arm][metric]:+.4f}"
                w.writerow({
                    "arm": arm,
                    "metric": metric,
                    "mean": f"{mean:.6f}",
                    "std": f"{std:.6f}",
                    "n_seeds": n,
                    "mean_pm_std": f"{mean:.4f} ± {std:.4f}",
                    "delta_vs_baseline": delta,
                    "baseline_arm": baseline_arm or "",
                })


def format_table(
    aggregated: dict[str, dict[str, tuple[float, float, int]]],
    metrics: list[str] | None = None,
    baseline_arm: str | None = None,
) -> str:
    """Fixed-width ``mean ± std`` table for the terminal / run README."""
    arms = sorted(aggregated)
    if not arms:
        return "(no runs)"
    if metrics is None:
        seen: list[str] = []
        for arm in arms:
            for k in aggregated[arm]:
                if k not in seen:
                    seen.append(k)
        metrics = seen
    width = max(len(m) for m in metrics) + 2
    # "0.1234±0.0567 (+0.0890)" is 23 chars; reserve for it so delta suffixes do
    # not run into the next column.
    cell_width = 25 if baseline_arm else 16
    col = max(max(len(a) for a in arms), cell_width) + 2
    lines = ["metric".ljust(width) + "".join(a.ljust(col) for a in arms)]
    lines.append("-" * (width + col * len(arms)))
    for m in metrics:
        row = m.ljust(width)
        for arm in arms:
            cell = aggregated[arm].get(m)
            if cell is None:
                row += "-".ljust(col)
                continue
            mean, std, _ = cell
            text = f"{mean:.4f}±{std:.4f}"
            if baseline_arm and arm != baseline_arm:
                base = aggregated.get(baseline_arm, {}).get(m)
                if base:
                    text += f" ({mean - base[0]:+.4f})"
            row += text.ljust(col)
        lines.append(row)
    return "\n".join(lines)
