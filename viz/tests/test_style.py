"""Tests for viz.src.style — style application and figure utilities."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

from viz.src.style import (
    apply_style,
    get_figure_size,
    get_method_color,
    get_method_marker,
    get_method_linestyle,
    get_cmap,
    add_demo_watermark,
    save_figure,
)


class TestApplyStyle:
    def test_applies_without_error(self) -> None:
        apply_style()
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False


class TestGetters:
    def test_figure_size_single(self) -> None:
        w, h = get_figure_size("single")
        assert w == 7.5
        assert h == 4.2

    def test_figure_size_unknown_falls_back(self) -> None:
        w, h = get_figure_size("nonexistent")
        assert w == 7.5  # fallback to single

    def test_method_color(self) -> None:
        assert get_method_color("FedAvg") == "#0072B2"
        assert get_method_color("FLOPS") == "#D55E00"

    def test_unknown_method_color(self) -> None:
        color = get_method_color("UnknownMethod")
        assert color == "#888888"

    def test_marker(self) -> None:
        assert get_method_marker("FedAvg") == "s"

    def test_linestyle(self) -> None:
        assert get_method_linestyle("FedProx") == "-."

    def test_cmap(self) -> None:
        assert get_cmap("class") == "Set3"


class TestWatermark:
    def test_adds_watermark(self) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        add_demo_watermark(fig)
        # Check that text was added
        texts = fig.texts
        assert len(texts) > 0
        assert "DEMO" in texts[0].get_text()
        plt.close(fig)


class TestSaveFigure:
    def test_save_creates_files(self, tmp_path: Path) -> None:
        apply_style()
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_title("Test")

        out_path = tmp_path / "test_fig"
        save_figure(fig, out_path, sidecar_data={"test": True})
        plt.close(fig)

        assert (tmp_path / "test_fig.png").exists()
        assert (tmp_path / "test_fig.pdf").exists()
        assert (tmp_path / "test_fig.json").exists()
        assert (tmp_path / "test_fig.png").stat().st_size > 0
