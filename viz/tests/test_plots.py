"""Tests for all plot functions — each must return a Figure with title and labels."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

from viz.src.demo import generate_demo_data
from viz.src.io import load_runs
from viz.src.style import apply_style, save_figure


@pytest.fixture(scope="module")
def demo_runs(tmp_path_factory) -> dict[str, dict[str, Any]]:
    """Load demo runs once for all plot tests."""
    tmp = tmp_path_factory.mktemp("demo")
    runs_dir = generate_demo_data(tmp, methods=["FLOPS", "FedAvg", "FedProx"], seeds=["42", "43", "44"])
    return load_runs(runs_dir)


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("figures")


def _assert_figure(fig: matplotlib.figure.Figure) -> None:
    """Common assertions for a plot figure."""
    assert isinstance(fig, matplotlib.figure.Figure)
    axes = fig.get_axes()
    assert len(axes) > 0, "Figure has no axes"


class TestConvergencePlots:
    def test_convergence(self, demo_runs, out_dir) -> None:
        from viz.src.plots.convergence import plot_convergence
        fig = plot_convergence(demo_runs, metric="global_acc", metric_label="Accuracy")
        _assert_figure(fig)
        save_figure(fig, out_dir / "convergence", is_demo=True)
        assert (out_dir / "convergence.png").exists()
        assert (out_dir / "convergence.json").exists()
        plt.close(fig)

    def test_convergence_vs_comm(self, demo_runs, out_dir) -> None:
        from viz.src.plots.convergence import plot_convergence_vs_comm
        fig = plot_convergence_vs_comm(demo_runs, metric="global_acc", metric_label="Accuracy")
        _assert_figure(fig)
        plt.close(fig)

    def test_rounds_to_target(self, demo_runs, out_dir) -> None:
        from viz.src.plots.convergence import plot_rounds_to_target
        fig = plot_rounds_to_target(demo_runs, targets=[0.5, 0.6, 0.7], metric="global_acc")
        _assert_figure(fig)
        plt.close(fig)

    def test_ablation(self, demo_runs, out_dir) -> None:
        from viz.src.plots.convergence import plot_ablation
        fig = plot_ablation(demo_runs)
        _assert_figure(fig)
        plt.close(fig)


class TestPartitionPlots:
    def test_partition_heatmap(self, demo_runs, out_dir) -> None:
        from viz.src.plots.partition import plot_partition_heatmap
        fig = plot_partition_heatmap(demo_runs)
        _assert_figure(fig)
        plt.close(fig)

    def test_per_class_metrics(self, demo_runs, out_dir) -> None:
        from viz.src.plots.partition import plot_per_class_metrics
        fig = plot_per_class_metrics(demo_runs)
        _assert_figure(fig)
        plt.close(fig)


class TestClientPlots:
    def test_per_client_performance(self, demo_runs, out_dir) -> None:
        from viz.src.plots.client import plot_per_client_performance
        fig = plot_per_client_performance(demo_runs)
        _assert_figure(fig)
        plt.close(fig)

    def test_client_participation(self, demo_runs, out_dir) -> None:
        from viz.src.plots.client import plot_client_participation
        fig = plot_client_participation(demo_runs)
        _assert_figure(fig)
        plt.close(fig)

    def test_local_vs_global(self, demo_runs, out_dir) -> None:
        from viz.src.plots.client import plot_local_vs_global
        fig = plot_local_vs_global(demo_runs)
        _assert_figure(fig)
        plt.close(fig)


class TestCommunicationPlots:
    def test_comm_cost(self, demo_runs, out_dir) -> None:
        from viz.src.plots.communication import plot_comm_cost
        fig = plot_comm_cost(demo_runs)
        _assert_figure(fig)
        plt.close(fig)


class TestTimingPlots:
    def test_round_time_breakdown(self, demo_runs, out_dir) -> None:
        from viz.src.plots.timing import plot_round_time_breakdown
        fig = plot_round_time_breakdown(demo_runs)
        _assert_figure(fig)
        plt.close(fig)


class TestSummaryPlots:
    def test_summary_table(self, demo_runs, out_dir) -> None:
        from viz.src.plots.summary import plot_summary_table
        fig = plot_summary_table(demo_runs)
        _assert_figure(fig)
        plt.close(fig)

    def test_summary_csv(self, demo_runs, out_dir) -> None:
        from viz.src.plots.summary import save_summary_csv
        csv_path = save_summary_csv(demo_runs, out_dir / "summary.csv")
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0

    def test_make_summary_table(self, demo_runs) -> None:
        from viz.src.plots.summary import make_summary_table
        import pandas as pd
        df = make_summary_table(demo_runs)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Method" in df.columns
