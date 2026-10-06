"""Partition and per-class visualizations (charts 4, 5)."""
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
    save_figure,
)

logger = logging.getLogger(__name__)


def plot_partition_heatmap(
    runs: dict[str, dict[str, Any]],
    setting: str = "",
) -> matplotlib.figure.Figure:
    """Chart 4: Client x Class sample-count heatmap.

    Uses data from the first seed of the first method (partition is shared).
    """
    apply_style()

    # Find partition data from any run
    partition_df = None
    for method, seeds in runs.items():
        for seed, data in seeds.items():
            if "partition" in data:
                partition_df = data["partition"]
                break
        if partition_df is not None:
            break

    if partition_df is None:
        logger.warning("No partition data found.")
        return plt.figure()

    pivot = partition_df.pivot_table(
        index="client_id", columns="class_id", values="count", fill_value=0
    )

    fig, ax = plt.subplots(figsize=get_figure_size("matrix"))
    im = ax.imshow(pivot.values, aspect="auto", cmap=get_cmap("diverging"))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Client ID")

    # Add text annotations
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            color = "white" if val > pivot.values.max() * 0.6 else "black"
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    fontsize=6, color=color)

    fig.colorbar(im, ax=ax, label="Sample Count", shrink=0.8)

    # Compute a simple skew metric: coefficient of variation across clients per class
    cv_per_class = pivot.std() / (pivot.mean() + 1e-8)
    skew_str = f"Mean CV={cv_per_class.mean():.2f}"

    title = f"Data Partition \u2014 Client \u00d7 Class Distribution ({skew_str})"
    if setting:
        title += f" ({setting})"
    ax.set_title(title, fontsize=10)

    # Turn off grid for heatmap
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_per_class_metrics(
    runs: dict[str, dict[str, Any]],
    metric: str = "f1",
    final_round: int | None = None,
    setting: str = "",
) -> matplotlib.figure.Figure:
    """Chart 5: Per-class recall/F1 at final round, highlighting minority classes."""
    apply_style()

    # Collect per_class data from first method, first seed
    pc_df = None
    method_name = None
    for method, seeds in runs.items():
        for seed, data in seeds.items():
            if "per_class" in data:
                pc_df = data["per_class"].copy()
                method_name = method
                break
        if pc_df is not None:
            break

    if pc_df is None:
        logger.warning("No per_class data found.")
        return plt.figure()

    if final_round is None:
        final_round = int(pc_df["round"].max())

    df_final = pc_df[pc_df["round"] == final_round].sort_values("class_id")

    fig, ax = plt.subplots(figsize=get_figure_size("single"))
    classes = df_final["class_id"].values
    values = df_final[metric].values
    support = df_final["support"].values

    # Color bars by support (minority classes highlighted)
    median_support = np.median(support)
    colors = ["#D55E00" if s < median_support * 0.5 else "#0072B2" for s in support]

    bars = ax.bar(classes, values, color=colors, edgecolor="white", linewidth=0.5)

    # Add support labels on top
    for bar, s in zip(bars, support):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"n={s}", ha="center", va="bottom", fontsize=6,
        )

    ax.set_ylim(0, min(1.15, max(values) + 0.1))
    ax.set_xlabel("Class ID")
    ax.set_ylabel(metric.upper())
    ax.set_xticks(classes)

    title = f"Per-Class {metric.upper()} \u2014 Round {final_round}"
    if setting:
        title += f" ({setting})"
    ax.set_title(title)

    # Legend for minority highlight
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#D55E00", label="Minority (support < 50% median)"),
        Patch(facecolor="#0072B2", label="Normal"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    return fig
