"""Tests for viz.src.io — schema validation and data loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from viz.src.io import SchemaError, validate_schema, load_run, load_runs
from viz.src.demo import generate_demo_data


@pytest.fixture
def demo_dir(tmp_path: Path) -> Path:
    """Generate demo data in a temp directory."""
    return generate_demo_data(tmp_path, methods=["FLOPS", "FedAvg"], seeds=["42", "43"])


class TestValidateSchema:
    """Tests for validate_schema()."""

    def test_valid_rounds_global(self) -> None:
        df = pd.DataFrame({
            "round": [1, 2, 3],
            "global_acc": [0.5, 0.6, 0.7],
            "global_loss": [1.0, 0.8, 0.6],
        })
        validate_schema(df, "rounds_global.csv", Path("/fake"))

    def test_missing_column(self) -> None:
        df = pd.DataFrame({
            "round": [1, 2, 3],
            "global_acc": [0.5, 0.6, 0.7],
        })
        with pytest.raises(SchemaError, match="missing columns"):
            validate_schema(df, "rounds_global.csv", Path("/fake"))

    def test_nan_values(self) -> None:
        df = pd.DataFrame({
            "round": [1, 2, 3],
            "global_acc": [0.5, np.nan, 0.7],
            "global_loss": [1.0, 0.8, 0.6],
        })
        with pytest.raises(SchemaError, match="NaN"):
            validate_schema(df, "rounds_global.csv", Path("/fake"))

    def test_non_continuous_rounds(self) -> None:
        df = pd.DataFrame({
            "round": [1, 2, 5],
            "global_acc": [0.5, 0.6, 0.7],
            "global_loss": [1.0, 0.8, 0.6],
        })
        with pytest.raises(SchemaError, match="non-continuous"):
            validate_schema(df, "rounds_global.csv", Path("/fake"))

    def test_wrong_dtype(self) -> None:
        df = pd.DataFrame({
            "round": ["a", "b", "c"],
            "global_acc": [0.5, 0.6, 0.7],
            "global_loss": [1.0, 0.8, 0.6],
        })
        with pytest.raises(SchemaError, match="expected integer"):
            validate_schema(df, "rounds_global.csv", Path("/fake"))


class TestLoadRun:
    """Tests for load_run()."""

    def test_load_single_run(self, demo_dir: Path) -> None:
        run = load_run(demo_dir / "FLOPS" / "42")
        assert "rounds_global" in run
        assert "config" in run
        assert isinstance(run["rounds_global"], pd.DataFrame)

    def test_missing_required_file(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Required file"):
            load_run(run_dir)


class TestLoadRuns:
    """Tests for load_runs()."""

    def test_load_all(self, demo_dir: Path) -> None:
        runs = load_runs(demo_dir)
        assert "FLOPS" in runs
        assert "FedAvg" in runs
        assert "42" in runs["FLOPS"]
        assert "43" in runs["FLOPS"]

    def test_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_runs(tmp_path / "nonexistent")
