"""Comparison plots — fills the `plots/` directory required by CLAUDE.md §21.

matplotlib is imported lazily with the Agg backend so this module stays importable
(and testable) on a headless machine with no display.

Design rules kept deliberately plain: no style sheets, no seaborn, deterministic
ordering, and every figure labels the number of seeds behind each bar so a reader
cannot mistake a 1-seed value for a 3-seed mean.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.bdd100k import TARGET_CLASSES


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_per_class_ap(
    aggregated: dict[str, dict[str, tuple[float, float, int]]],
    out_path: Path,
    metric_prefix: str = "AP50",
    title: str | None = None,
) -> Path | None:
    """Grouped bars: per-class AP per arm, with std error bars.

    Returns the written path, or None when no arm has the requested metric.
    """
    arms = sorted(aggregated)
    classes = [c for c in TARGET_CLASSES
               if any(f"{metric_prefix}_{c}" in aggregated[a] for a in arms)]
    if not arms or not classes:
        return None

    plt = _plt()
    n_arms = len(arms)
    width = 0.8 / n_arms
    fig, ax = plt.subplots(figsize=(1.9 * len(classes) + 3, 4.2))

    for i, arm in enumerate(arms):
        xs, means, errs = [], [], []
        for j, cls in enumerate(classes):
            cell = aggregated[arm].get(f"{metric_prefix}_{cls}")
            if cell is None:
                continue
            xs.append(j - 0.4 + width * (i + 0.5))
            means.append(cell[0])
            errs.append(cell[1])
        n_seeds = {aggregated[arm][f"{metric_prefix}_{c}"][2]
                   for c in classes if f"{metric_prefix}_{c}" in aggregated[arm]}
        label = f"{arm} (n={'/'.join(str(n) for n in sorted(n_seeds))})"
        ax.bar(xs, means, width=width, yerr=errs, capsize=3, label=label)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel(metric_prefix)
    ax.set_title(title or f"Per-class {metric_prefix} (mean ± std, final round)")
    ax.legend(fontsize="small")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_metric_over_rounds(
    records: list[Any],
    metric: str,
    out_path: Path,
    title: str | None = None,
) -> Path | None:
    """Metric vs round, one line per arm (mean over seeds at each round).

    ``records`` is a list of ``compare.RunRecord``. Rounds present for only some
    seeds are averaged over whatever seeds reached them, which is why the legend
    shows the seed count.
    """
    arms = sorted({r.arm for r in records})
    series: dict[str, tuple[list[int], list[float], int]] = {}
    for arm in arms:
        arm_recs = [r for r in records if r.arm == arm]
        per_round: dict[int, list[float]] = {}
        for r in arm_recs:
            for row in r.rounds:
                if metric in row:
                    per_round.setdefault(int(row["round"]), []).append(row[metric])
        if not per_round:
            continue
        rounds = sorted(per_round)
        series[arm] = (
            rounds,
            [sum(per_round[x]) / len(per_round[x]) for x in rounds],
            len(arm_recs),
        )
    if not series:
        return None

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for arm in sorted(series):
        rounds, values, n = series[arm]
        ax.plot(rounds, values, marker="o", markersize=3, label=f"{arm} (n={n})")
    ax.set_xlabel("round (absolute)")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} over rounds (mean over seeds)")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_delta_bars(
    deltas: dict[str, dict[str, float]],
    out_path: Path,
    metric_prefix: str = "AP50",
    baseline_arm: str = "A0",
) -> Path | None:
    """ΔAP_c vs the baseline, per class per arm (CLAUDE.md §13).

    Classes where an arm is WORSE show as negative bars — they must stay visible,
    since hiding classes where the method loses is explicitly forbidden (§20).
    """
    arms = sorted(deltas)
    classes = [c for c in TARGET_CLASSES
               if any(f"{metric_prefix}_{c}" in deltas[a] for a in arms)]
    if not arms or not classes:
        return None

    plt = _plt()
    width = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(1.9 * len(classes) + 3, 4.2))
    for i, arm in enumerate(arms):
        xs, vals = [], []
        for j, cls in enumerate(classes):
            key = f"{metric_prefix}_{cls}"
            if key not in deltas[arm]:
                continue
            xs.append(j - 0.4 + width * (i + 0.5))
            vals.append(deltas[arm][key])
        ax.bar(xs, vals, width=width, label=arm)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_ylabel(f"Δ{metric_prefix} vs {baseline_arm}")
    ax.set_title(f"Δ{metric_prefix} per class vs {baseline_arm} (final round)")
    ax.legend(fontsize="small")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
