from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.data.bdd100k import TARGET_CLASSES


def flatten_per_class_metrics(raw: dict[str, float]) -> dict[str, Any]:
    """Reshape ultralytics val() output into per-class rows."""
    rows = []
    for cls in TARGET_CLASSES:
        row = {"class": cls}
        for metric in ("AP50", "AP", "precision", "recall"):
            key = f"{metric}_{cls}"
            row[metric] = raw.get(key, None)
        rows.append(row)
    return rows


def save_per_class_metrics(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def compute_delta_ap(
    method_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> dict[str, float]:
    """ΔAP_c = AP_method,c - AP_FedAvg,c (Section 13, CLAUDE.md)."""
    return {
        f"delta_AP50_{cls}": (
            method_metrics.get(f"AP50_{cls}", 0.0)
            - baseline_metrics.get(f"AP50_{cls}", 0.0)
        )
        for cls in TARGET_CLASSES
    }


def save_summary_metrics(
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for k, v in sorted(metrics.items()):
            writer.writerow({"metric": k, "value": v})
