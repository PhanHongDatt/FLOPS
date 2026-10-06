"""Summary table and FL ops timeline (charts 13, 14)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz.src.io import aggregate_over_seeds
from viz.src.style import (
    apply_style,
    get_figure_size,
    get_method_color,
)

logger = logging.getLogger(__name__)


def make_summary_table(
    runs: dict[str, dict[str, Any]],
    metric: str = "global_acc",
    metric_label: str = "Accuracy",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Compute summary table: best and final metric +/- std per method."""
    agg = aggregate_over_seeds(runs, "rounds_global", metric)
    if agg.empty:
        return pd.DataFrame()

    records = []
    for method in agg["method"].unique():
        mdf = agg[agg["method"] == method].sort_values("round")
        # Final round
        final = mdf.iloc[-1]
        # Best round
        if higher_is_better:
            best = mdf.loc[mdf["mean"].idxmax()]
        else:
            best = mdf.loc[mdf["mean"].idxmin()]

        records.append({
            "Method": method,
            f"Final {metric_label}": f"{final['mean']:.4f} \u00b1 {final['std']:.4f}",
            "Final Round": int(final["round"]),
            f"Best {metric_label}": f"{best['mean']:.4f} \u00b1 {best['std']:.4f}",
            "Best Round": int(best["round"]),
            "Seeds": int(final["n_seeds"]),
        })

    return pd.DataFrame(records)


def plot_summary_table(
    runs: dict[str, dict[str, Any]],
    metric: str = "global_acc",
    metric_label: str = "Accuracy",
    higher_is_better: bool = True,
    setting: str = "",
) -> matplotlib.figure.Figure:
    """Chart 13: Summary table rendered as a figure."""
    apply_style()
    df = make_summary_table(runs, metric, metric_label, higher_is_better)

    if df.empty:
        logger.warning("No data for summary table.")
        return plt.figure()

    fig, ax = plt.subplots(
        figsize=(max(7.5, len(df.columns) * 1.8), max(2.0, len(df) * 0.5 + 1.0))
    )
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Color method cells
    for i, method in enumerate(df["Method"]):
        table[i + 1, 0].set_facecolor(get_method_color(method))
        table[i + 1, 0].set_text_props(color="white", fontweight="bold")

    # Header style
    for j in range(len(df.columns)):
        table[0, j].set_facecolor("#333333")
        table[0, j].set_text_props(color="white", fontweight="bold")

    title = f"Summary \u2014 {metric_label} Comparison"
    if setting:
        title += f" ({setting})"
    ax.set_title(title, fontsize=12, pad=20)

    fig.tight_layout()
    return fig


def save_summary_csv(
    runs: dict[str, dict[str, Any]],
    out_path: Path,
    metric: str = "global_acc",
    metric_label: str = "Accuracy",
    higher_is_better: bool = True,
) -> Path:
    """Export summary table as CSV."""
    df = make_summary_table(runs, metric, metric_label, higher_is_better)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved summary CSV to %s", out_path)
    return out_path
