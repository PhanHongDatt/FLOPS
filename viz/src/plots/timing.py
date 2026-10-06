"""System and timing plots (chart 10)."""
from __future__ import annotations

import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz.src.style import (
    apply_style,
    get_figure_size,
)

logger = logging.getLogger(__name__)


def plot_round_time_breakdown(
    runs: dict[str, dict[str, Any]],
    setting: str = "",
) -> matplotlib.figure.Figure:
    """Chart 10: Round time breakdown with straggler highlights."""
    apply_style()

    # Find system data
    sys_df = None
    rc_df = None
    method_name = None
    for method, seeds in runs.items():
        for seed, data in seeds.items():
            if "system" in data:
                sys_df = data["system"].copy()
                method_name = method
            if "rounds_client" in data:
                rc_df = data["rounds_client"]
            break
        if sys_df is not None:
            break

    if sys_df is None:
        logger.warning("No system data for round_time_breakdown.")
        return plt.figure()

    fig, axes = plt.subplots(2, 1, figsize=get_figure_size("single"),
                              gridspec_kw={"height_ratios": [3, 1]})

    # Top: round time with straggler markers
    ax_top = axes[0]
    rounds = sys_df["round"].values
    times = sys_df["round_time_s"].values

    ax_top.bar(rounds, times, color="#0072B2", alpha=0.7, width=0.8)

    # Highlight straggler rounds
    if "stragglers" in sys_df.columns:
        straggler_mask = sys_df["stragglers"] > 0
        if straggler_mask.any():
            strag_rounds = sys_df.loc[straggler_mask, "round"].values
            strag_times = sys_df.loc[straggler_mask, "round_time_s"].values
            ax_top.bar(strag_rounds, strag_times, color="#D55E00",
                       alpha=0.8, width=0.8, label="Has straggler(s)")
            ax_top.legend(loc="upper right", framealpha=0.8)

    ax_top.set_ylabel("Round Time (s)")
    ax_top.set_ylim(bottom=0)

    title = f"Round Time Breakdown \u2014 Per-Round Duration"
    if setting:
        title += f" ({setting})"
    ax_top.set_title(title)

    # Bottom: resource usage
    ax_bot = axes[1]
    if "cpu_pct" in sys_df.columns:
        ax_bot.plot(rounds, sys_df["cpu_pct"], color="#009E73",
                    linewidth=1.0, label="CPU %")
        ax_bot.set_ylabel("CPU %")
    if "mem_mb" in sys_df.columns:
        ax2 = ax_bot.twinx()
        ax2.plot(rounds, sys_df["mem_mb"], color="#CC79A7",
                 linewidth=1.0, linestyle="--", label="Mem MB")
        ax2.set_ylabel("Memory (MB)")

    ax_bot.set_xlabel("Round")
    ax_bot.legend(loc="upper left", framealpha=0.8, fontsize=7)

    fig.tight_layout()
    return fig
