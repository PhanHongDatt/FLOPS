"""Style configuration and figure utilities for FL visualization."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

_STYLE_CFG: dict[str, Any] | None = None
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "style.yaml"


def _load_style() -> dict[str, Any]:
    global _STYLE_CFG
    if _STYLE_CFG is None:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _STYLE_CFG = yaml.safe_load(f)
        logger.info("Loaded style config from %s", _CONFIG_PATH)
    return _STYLE_CFG


def apply_style() -> None:
    """Apply global matplotlib style from style.yaml."""
    cfg = _load_style()
    font = cfg["font"]
    plt.rcParams.update({
        "font.family": font["family"],
        "axes.titlesize": font["title_size"],
        "axes.labelsize": font["label_size"],
        "xtick.labelsize": font["tick_size"],
        "ytick.labelsize": font["tick_size"],
        "legend.fontsize": font["legend_size"],
        "axes.spines.top": cfg["spines"]["top"],
        "axes.spines.right": cfg["spines"]["right"],
        "axes.grid": cfg["grid"]["enabled"],
        "grid.alpha": cfg["grid"]["alpha"],
        "grid.linestyle": cfg["grid"]["linestyle"],
        "figure.dpi": cfg["figure"]["dpi"],
        "savefig.dpi": cfg["figure"]["dpi"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })
    logger.info("Applied FL viz style.")


def get_figure_size(preset: str = "single") -> tuple[float, float]:
    """Return (width, height) for a figure preset."""
    cfg = _load_style()
    sizes = cfg["figure"]
    if preset not in sizes:
        logger.warning("Unknown preset '%s', falling back to 'single'.", preset)
        preset = "single"
    w, h = sizes[preset]
    return (float(w), float(h))


def get_method_color(method: str) -> str:
    """Return the fixed color for a method."""
    cfg = _load_style()
    colors = cfg["palette"]["method"]
    if method not in colors:
        logger.warning("No color for method '%s', using grey.", method)
        return "#888888"
    return colors[method]


def get_method_marker(method: str) -> str:
    """Return the marker for a method."""
    cfg = _load_style()
    markers = cfg["markers"]
    return markers.get(method, "o")


def get_method_linestyle(method: str) -> str:
    """Return the linestyle for a method."""
    cfg = _load_style()
    ls = cfg["linestyles"]
    return ls.get(method, "-")


def get_cmap(kind: str) -> str:
    """Return colormap name for 'class', 'client', or 'diverging'."""
    cfg = _load_style()
    key = f"{kind}_cmap"
    return cfg["palette"].get(key, "viridis")


def get_output_formats() -> list[str]:
    """Return list of output formats (e.g. ['png', 'pdf'])."""
    cfg = _load_style()
    return cfg["figure"].get("formats", ["png"])


def add_demo_watermark(fig: matplotlib.figure.Figure) -> None:
    """Stamp 'DEMO DATA' watermark across the figure."""
    cfg = _load_style()
    wm = cfg["demo_watermark"]
    fig.text(
        0.5, 0.5, wm["text"],
        fontsize=wm["fontsize"],
        color=wm["color"],
        alpha=wm["alpha"],
        ha="center", va="center",
        rotation=wm["rotation"],
        transform=fig.transFigure,
        zorder=999,
    )


def save_figure(
    fig: matplotlib.figure.Figure,
    path: Path,
    *,
    sidecar_data: dict[str, Any] | None = None,
    is_demo: bool = False,
) -> None:
    """Save figure in all configured formats and write sidecar JSON."""
    if is_demo:
        add_demo_watermark(fig)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in get_output_formats():
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out)
        logger.info("Saved %s", out)

    # Sidecar JSON
    sidecar = sidecar_data.copy() if sidecar_data else {}
    sidecar["created_at"] = datetime.now(timezone.utc).isoformat()

    # Hash the PNG for reproducibility tracking
    png_path = path.with_suffix(".png")
    if png_path.exists():
        h = hashlib.sha256(png_path.read_bytes()).hexdigest()[:16]
        sidecar["png_sha256_prefix"] = h

    json_path = path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, default=str)
    logger.info("Wrote sidecar %s", json_path)
