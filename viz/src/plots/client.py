"""Client-centric plots (charts 6, 7, 8)."""
from __future__ import annotations

import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz.src.style import (
    apply_style,
    get_cmap,
    get_figure_size,
    get_method_color,
)

logger = logging.getLogger(__name__)


def plot_per_client_performance(
    runs: dict[str, dict[str, Any]],
    metric_col: str = "local_metric",
    metric_label: str = "Local Metric",
    setting: str = "",
) -> matplotlib.figure.Figure:
    """Chart 6: Per-client metric distribution (boxplot) at final round."""
    apply_style()

    # Collect from first seed of each method
    all_data = []
    for method, seeds in runs.items():
        for seed, data in seeds.items():
            if "rounds_client" not in data:
                continue
            rc = data["rounds_client"]
            final_round = rc["round"].max()
            final = rc[(rc["round"] == final_round) & (rc["participated"])]
            for _, row in final.iterrows():
                all_data.append({
                    "method": method,
                    "client_id": row["client_id"],
                    metric_col: row[metric_col],
                })
            break  # first seed only

    if not all_data:
        logger.warning("No client data for per_client_performance.")
        return plt.figure()

    df = pd.DataFrame(all_data)
    methods = sorted(df["method"].unique())

    fig, ax = plt.subplots(figsize=get_figure_size("single"))

    data_arrays = [df[df["method"] == m][metric_col].values for m in methods]
    bp = ax.boxplot(
        data_arrays,
        labels=methods,
        patch_artist=True,
        widths=0.5,
    )

    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(get_method_color(method))
        patch.set_alpha(0.6)

    # Annotate stats
    for i, (arr, method) in enumerate(zip(data_arrays, methods)):
        if len(arr) > 0:
            stats_text = f"min={arr.min():.3f}\nmed={np.median(arr):.3f}\nstd={arr.std():.3f}"
            ax.text(
                i + 1, arr.min() - 0.02, stats_text,
                ha="center", va="top", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

    title = f"Client Performance \u2014 {metric_label} Distribution"
    if setting:
        title += f" ({setting})"
    ax.set_title(title)
    ax.set_ylabel(metric_label)
    fig.tight_layout()
    return fig


def plot_client_participation(
    runs: dict[str, dict[str, Any]],
    setting: str = "",
) -> matplotlib.figure.Figure:
    """Chart 7: Client participation heatmap (round x client)."""
    apply_style()

    # Use first method, first seed
    rc_df = None
    for method, seeds in runs.items():
        for seed, data in seeds.items():
            if "rounds_client" in data:
                rc_df = data["rounds_client"]
                break
        if rc_df is not None:
            break

    if rc_df is None:
        logger.warning("No rounds_client data for participation heatmap.")
        return plt.figure()

    pivot = rc_df.pivot_table(
        index="round", columns="client_id",
        values="participated", aggfunc="first",
    ).fillna(False).astype(int)

    fig, ax = plt.subplots(figsize=get_figure_size("wide"))
    im = ax.imshow(
        pivot.values.T, aspect="auto", cmap="YlGn",
        interpolation="nearest",
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Client")
    ax.set_yticks(range(len(pivot.columns)))
    ax.set_yticklabels(pivot.columns, fontsize=7)

    # Show every 10th round on x-axis
    xticks = list(range(0, len(pivot.index), max(1, len(pivot.index) // 10)))
    ax.set_xticks(xticks)
    ax.set_xticklabels([pivot.index[i] for i in xticks])

    fig.colorbar(im, ax=ax, label="Participated", shrink=0.6)

    title = f"Client Participation \u2014 Round \u00d7 Client"
    if setting:
        title += f" ({setting})"
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_local_vs_global(
    runs: dict[str, dict[str, Any]],
    metric: str = "global_acc",
    metric_label: str = "Accuracy",
    setting: str = "",
) -> matplotlib.figure.Figure:
    """Chart 8: Local metric vs global metric over rounds (drift detection)."""
    apply_style()

    fig, ax = plt.subplots(figsize=get_figure_size("single"))
    plotted = False

    for method, seeds in runs.items():
        for seed, data in seeds.items():
            if "rounds_global" not in data or "rounds_client" not in data:
                continue

            rg = data["rounds_global"].sort_values("round")
            rc = data["rounds_client"]

            # Mean local metric per round (participating clients only)
            local_mean = (
                rc[rc["participated"]]
                .groupby("round")["local_metric"]
                .mean()
                .reset_index()
                .rename(columns={"local_metric": "local_mean"})
            )

            color = get_method_color(method)
            ax.plot(
                rg["round"], rg[metric],
                color=color, linestyle="-", linewidth=1.5,
                label=f"{method} global",
            )
            ax.plot(
                local_mean["round"], local_mean["local_mean"],
                color=color, linestyle="--", linewidth=1.0, alpha=0.7,
                label=f"{method} local mean",
            )
            plotted = True
            break  # first seed

    if not plotted:
        logger.warning("No data for local_vs_global plot.")
        return fig

    title = f"Local vs Global \u2014 {metric_label} (Drift Detection)"
    if setting:
        title += f" ({setting})"
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(metric_label)
    ax.legend(loc="best", framealpha=0.8, fontsize=7)
    fig.tight_layout()
    return fig
