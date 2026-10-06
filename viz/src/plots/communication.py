"""Communication cost plots (chart 9)."""
from __future__ import annotations

import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz.src.io import aggregate_client_comm
from viz.src.style import (
    apply_style,
    get_figure_size,
    get_method_color,
    get_method_linestyle,
    get_method_marker,
)

logger = logging.getLogger(__name__)


def plot_comm_cost(
    runs: dict[str, dict[str, Any]],
    setting: str = "",
) -> matplotlib.figure.Figure:
    """Chart 9: Cumulative communication cost per method."""
    apply_style()
    agg = aggregate_client_comm(runs)

    if agg.empty:
        logger.warning("No data for comm_cost plot.")
        return plt.figure()

    fig, axes = plt.subplots(1, 2, figsize=get_figure_size("dual"))

    # Left: per-round bytes
    ax_left = axes[0]
    for method, seeds in runs.items():
        for seed, data in seeds.items():
            if "rounds_client" not in data:
                continue
            rc = data["rounds_client"]
            per_round = rc.groupby("round").agg(
                bytes_up_total=("bytes_up", "sum"),
                bytes_down_total=("bytes_down", "sum"),
            ).reset_index()
            per_round["total_mb"] = (
                per_round["bytes_up_total"] + per_round["bytes_down_total"]
            ) / 1e6

            color = get_method_color(method)
            ax_left.plot(
                per_round["round"], per_round["total_mb"],
                color=color, linewidth=1.2, alpha=0.7,
                label=method,
            )
            break  # first seed

    ax_left.set_xlabel("Round")
    ax_left.set_ylabel("Communication per Round (MB)")
    ax_left.set_title("Per-Round Communication")
    ax_left.legend(loc="best", framealpha=0.8)
    ax_left.set_ylim(bottom=0)

    # Right: cumulative
    ax_right = axes[1]
    for method in agg["method"].unique():
        mdf = agg[agg["method"] == method].sort_values("round")
        color = get_method_color(method)
        ax_right.plot(
            mdf["round"], mdf["cum_mb_mean"],
            color=color,
            linestyle=get_method_linestyle(method),
            marker=get_method_marker(method),
            markevery=max(1, len(mdf) // 10),
            label=method,
            linewidth=1.8,
        )
        if "cum_mb_std" in mdf.columns:
            ax_right.fill_between(
                mdf["round"],
                mdf["cum_mb_mean"] - mdf["cum_mb_std"],
                mdf["cum_mb_mean"] + mdf["cum_mb_std"],
                color=color, alpha=0.1,
            )

    ax_right.set_xlabel("Round")
    ax_right.set_ylabel("Cumulative Communication (MB)")
    ax_right.set_title("Cumulative Communication")
    ax_right.legend(loc="best", framealpha=0.8)
    ax_right.set_ylim(bottom=0)

    title = f"Communication Cost \u2014 Up + Down"
    if setting:
        title += f" ({setting})"
    fig.suptitle(title)
    fig.tight_layout()
    return fig
