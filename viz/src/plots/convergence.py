"""Convergence and target-based plots (charts 1, 2, 3, 11, 12)."""
from __future__ import annotations

import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz.src.io import aggregate_over_seeds, aggregate_client_comm
from viz.src.style import (
    apply_style,
    get_figure_size,
    get_method_color,
    get_method_linestyle,
    get_method_marker,
    save_figure,
)

logger = logging.getLogger(__name__)


def plot_convergence(
    runs: dict[str, dict[str, Any]],
    metric: str = "global_acc",
    metric_label: str = "Accuracy",
    setting: str = "",
    confidence: float = 0.95,
    higher_is_better: bool = True,
) -> matplotlib.figure.Figure:
    """Chart 1: Global metric vs round, mean +/- CI per method."""
    apply_style()
    agg = aggregate_over_seeds(runs, "rounds_global", metric, confidence=confidence)
    if agg.empty:
        logger.warning("No data for convergence plot.")
        return plt.figure()

    fig, ax = plt.subplots(figsize=get_figure_size("single"))

    for method in agg["method"].unique():
        mdf = agg[agg["method"] == method].sort_values("round")
        color = get_method_color(method)
        ax.plot(
            mdf["round"], mdf["mean"],
            color=color,
            linestyle=get_method_linestyle(method),
            marker=get_method_marker(method),
            markevery=max(1, len(mdf) // 10),
            label=f"{method} (n={mdf['n_seeds'].iloc[0]})",
            linewidth=1.8,
            markersize=5,
        )
        ax.fill_between(
            mdf["round"], mdf["ci_lo"], mdf["ci_hi"],
            color=color, alpha=0.15,
        )

    title = f"Convergence \u2014 {metric_label} vs Round"
    if setting:
        title += f" ({setting})"
    ax.set_title(title)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel(metric_label)
    ax.legend(loc="best", framealpha=0.8)
    fig.tight_layout()
    return fig


def plot_convergence_vs_comm(
    runs: dict[str, dict[str, Any]],
    metric: str = "global_acc",
    metric_label: str = "Accuracy",
    setting: str = "",
    confidence: float = 0.95,
) -> matplotlib.figure.Figure:
    """Chart 2: Global metric vs cumulative communication (MB)."""
    apply_style()
    agg_metric = aggregate_over_seeds(runs, "rounds_global", metric, confidence=confidence)
    agg_comm = aggregate_client_comm(runs)

    if agg_metric.empty or agg_comm.empty:
        logger.warning("No data for convergence_vs_comm plot.")
        return plt.figure()

    fig, ax = plt.subplots(figsize=get_figure_size("single"))

    for method in agg_metric["method"].unique():
        mdf = agg_metric[agg_metric["method"] == method].sort_values("round")
        cdf = agg_comm[agg_comm["method"] == method].sort_values("round")

        # Merge on round
        merged = mdf.merge(cdf, on=["method", "round"], suffixes=("_metric", "_comm"))
        if merged.empty:
            continue

        color = get_method_color(method)
        ax.plot(
            merged["cum_mb_mean"], merged["mean"],
            color=color,
            linestyle=get_method_linestyle(method),
            marker=get_method_marker(method),
            markevery=max(1, len(merged) // 10),
            label=method,
            linewidth=1.8,
            markersize=5,
        )
        ax.fill_between(
            merged["cum_mb_mean"],
            merged["ci_lo"], merged["ci_hi"],
            color=color, alpha=0.15,
        )

    title = f"Convergence vs Communication \u2014 {metric_label} vs Cumulative MB"
    if setting:
        title += f" ({setting})"
    ax.set_title(title)
    ax.set_xlabel("Cumulative Communication (MB, up+down)")
    ax.set_ylabel(metric_label)
    ax.legend(loc="best", framealpha=0.8)
    fig.tight_layout()
    return fig


def plot_rounds_to_target(
    runs: dict[str, dict[str, Any]],
    targets: list[float],
    metric: str = "global_acc",
    metric_label: str = "Accuracy",
    setting: str = "",
    higher_is_better: bool = True,
) -> matplotlib.figure.Figure:
    """Chart 3: Rounds (and MB) to reach target metric thresholds."""
    apply_style()
    agg = aggregate_over_seeds(runs, "rounds_global", metric)
    agg_comm = aggregate_client_comm(runs)

    if agg.empty:
        logger.warning("No data for rounds_to_target plot.")
        return plt.figure()

    methods = sorted(agg["method"].unique())
    records = []

    for method in methods:
        mdf = agg[agg["method"] == method].sort_values("round")
        cdf = agg_comm[agg_comm["method"] == method].sort_values("round") if not agg_comm.empty else pd.DataFrame()

        for target in targets:
            if higher_is_better:
                reached = mdf[mdf["mean"] >= target]
            else:
                reached = mdf[mdf["mean"] <= target]

            if reached.empty:
                records.append({"method": method, "target": target, "round": np.nan, "mb": np.nan})
            else:
                rnd = int(reached["round"].iloc[0])
                mb = np.nan
                if not cdf.empty:
                    cm = cdf[cdf["round"] == rnd]
                    if not cm.empty:
                        mb = float(cm["cum_mb_mean"].iloc[0])
                records.append({"method": method, "target": target, "round": rnd, "mb": mb})

    rdf = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=get_figure_size("dual"))

    for i, (col, ylabel) in enumerate([("round", "Rounds"), ("mb", "Cumulative MB")]):
        ax = axes[i]
        x = np.arange(len(targets))
        width = 0.8 / max(len(methods), 1)

        for j, method in enumerate(methods):
            vals = []
            for target in targets:
                row = rdf[(rdf["method"] == method) & (rdf["target"] == target)]
                vals.append(float(row[col].iloc[0]) if not row.empty and pd.notna(row[col].iloc[0]) else 0)
            ax.bar(
                x + j * width - 0.4 + width / 2,
                vals,
                width=width * 0.9,
                color=get_method_color(method),
                label=method if i == 0 else None,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.0%}" for t in targets])
        ax.set_xlabel(f"Target {metric_label}")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)

    title = f"Rounds to Target \u2014 {metric_label}"
    if setting:
        title += f" ({setting})"
    fig.suptitle(title)
    axes[0].legend(loc="best", framealpha=0.8)
    fig.tight_layout()
    return fig


def plot_ablation(
    runs: dict[str, dict[str, Any]],
    metric: str = "global_acc",
    metric_label: str = "Accuracy",
    setting: str = "",
    confidence: float = 0.95,
) -> matplotlib.figure.Figure:
    """Chart 11: Ablation variants on same scale (reuses convergence logic)."""
    return plot_convergence(
        runs, metric=metric, metric_label=f"{metric_label} (Ablation)",
        setting=setting, confidence=confidence,
    )


def plot_sensitivity(
    data: pd.DataFrame,
    x_col: str,
    y_col: str = "mean",
    err_col: str = "std",
    x_label: str = "",
    y_label: str = "Metric",
    setting: str = "",
) -> matplotlib.figure.Figure:
    """Chart 12: Metric vs hyperparameter with error bars.

    Expects a DataFrame with columns: <x_col>, <y_col>, <err_col>, method.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=get_figure_size("single"))

    for method in data["method"].unique():
        mdf = data[data["method"] == method].sort_values(x_col)
        color = get_method_color(method)
        ax.errorbar(
            mdf[x_col], mdf[y_col], yerr=mdf[err_col],
            color=color,
            marker=get_method_marker(method),
            linestyle=get_method_linestyle(method),
            label=method,
            capsize=3,
            linewidth=1.5,
        )

    title = f"Sensitivity \u2014 {y_label} vs {x_label or x_col}"
    if setting:
        title += f" ({setting})"
    ax.set_title(title)
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label)
    ax.legend(loc="best", framealpha=0.8)
    fig.tight_layout()
    return fig
