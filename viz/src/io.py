"""Data loading, schema validation, and seed aggregation for FL logs."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------- schema definitions ----------

_SCHEMAS: dict[str, dict[str, type]] = {
    "rounds_global.csv": {
        "round": np.integer,
        "global_acc": np.floating,
        "global_loss": np.floating,
    },
    "rounds_client.csv": {
        "round": np.integer,
        "client_id": object,         # string or int
        "participated": np.bool_,
        "num_samples": np.integer,
        "local_loss": np.floating,
        "local_metric": np.floating,
        "train_time_s": np.floating,
        "bytes_up": np.integer,
        "bytes_down": np.integer,
    },
    "partition.csv": {
        "client_id": object,
        "class_id": np.integer,
        "count": np.integer,
    },
    "per_class.csv": {
        "round": np.integer,
        "class_id": np.integer,
        "precision": np.floating,
        "recall": np.floating,
        "f1": np.floating,
        "support": np.integer,
    },
    "system.csv": {
        "round": np.integer,
        "round_time_s": np.floating,
        "stragglers": np.integer,
        "cpu_pct": np.floating,
        "mem_mb": np.floating,
    },
}

_REQUIRED_FILES = {"rounds_global.csv"}
_OPTIONAL_FILES = {"rounds_client.csv", "partition.csv", "per_class.csv", "system.csv"}


# ---------- validation ----------

class SchemaError(Exception):
    """Raised when a CSV file does not match the expected schema."""


def validate_schema(df: pd.DataFrame, filename: str, run_path: Path) -> None:
    """Validate column presence, types, NaN, and round continuity."""
    schema = _SCHEMAS.get(filename)
    if schema is None:
        return  # no schema to validate

    context = f"{run_path / filename}"

    # Check required columns
    missing = set(schema.keys()) - set(df.columns)
    if missing:
        raise SchemaError(f"{context}: missing columns {missing}")

    # Check NaN in required columns
    for col in schema:
        if df[col].isna().any():
            raise SchemaError(
                f"{context}: column '{col}' contains NaN values"
            )

    # Check types (numeric vs object)
    for col, expected_type in schema.items():
        if expected_type is object:
            continue
        if expected_type in (np.bool_,):
            # Accept bool-like columns
            if not pd.api.types.is_bool_dtype(df[col]):
                # Try to cast
                try:
                    df[col] = df[col].astype(bool)
                except (ValueError, TypeError) as e:
                    raise SchemaError(
                        f"{context}: column '{col}' cannot be cast to bool: {e}"
                    )
        elif issubclass(expected_type, np.integer):
            if not pd.api.types.is_integer_dtype(df[col]):
                raise SchemaError(
                    f"{context}: column '{col}' expected integer, got {df[col].dtype}"
                )
        elif issubclass(expected_type, np.floating):
            if not pd.api.types.is_float_dtype(df[col]):
                # Try numeric conversion
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise SchemaError(
                        f"{context}: column '{col}' expected float, got {df[col].dtype}"
                    )

    # Check round continuity if 'round' column exists
    if "round" in df.columns:
        rounds = sorted(df["round"].unique())
        expected = list(range(rounds[0], rounds[-1] + 1))
        if rounds != expected:
            missing_rounds = set(expected) - set(rounds)
            raise SchemaError(
                f"{context}: non-continuous rounds, missing: {missing_rounds}"
            )

    logger.debug("Schema OK: %s", context)


# ---------- loading ----------

def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with standard options."""
    return pd.read_csv(path)


def load_run(run_dir: Path) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Load all data files from a single run directory.

    Returns dict mapping filename (without .csv) to DataFrame,
    plus 'config' -> dict from config.yaml.
    """
    result: dict[str, Any] = {}

    # Load config
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            result["config"] = yaml.safe_load(f)
    else:
        logger.warning("No config.yaml in %s", run_dir)
        result["config"] = {}

    # Load required files
    for filename in _REQUIRED_FILES:
        path = run_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )
        df = _read_csv(path)
        validate_schema(df, filename, run_dir)
        result[filename.replace(".csv", "")] = df

    # Load optional files
    for filename in _OPTIONAL_FILES:
        path = run_dir / filename
        if path.exists():
            df = _read_csv(path)
            validate_schema(df, filename, run_dir)
            result[filename.replace(".csv", "")] = df
        else:
            logger.info("Optional file %s not found in %s, skipping.", filename, run_dir)

    return result


def load_runs(runs_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all runs organized as runs/<method>/<seed>/.

    Returns nested dict: {method: {seed: {filename: DataFrame, ...}}}.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    result: dict[str, dict[str, Any]] = {}

    for method_dir in sorted(runs_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        result[method] = {}

        for seed_dir in sorted(method_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            seed = seed_dir.name
            try:
                result[method][seed] = load_run(seed_dir)
                logger.info("Loaded run: %s / %s", method, seed)
            except (SchemaError, FileNotFoundError) as e:
                logger.error("Failed to load %s/%s: %s", method, seed, e)
                raise

    if not result:
        raise FileNotFoundError(f"No method directories found in {runs_dir}")

    return result


# ---------- aggregation over seeds ----------

def aggregate_over_seeds(
    runs: dict[str, dict[str, Any]],
    table: str,
    metric: str,
    group_col: str = "round",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Aggregate a metric across seeds for each method.

    Returns DataFrame with columns:
        method, <group_col>, mean, std, ci_lo, ci_hi, n_seeds
    """
    rows = []
    for method, seeds in runs.items():
        # Collect per-seed series
        seed_dfs = []
        for seed, data in seeds.items():
            if table not in data:
                logger.warning("%s/%s missing table '%s'", method, seed, table)
                continue
            df = data[table][[group_col, metric]].copy()
            df = df.rename(columns={metric: f"value_{seed}"})
            seed_dfs.append(df)

        if not seed_dfs:
            continue

        # Merge on group_col
        merged = seed_dfs[0]
        for sdf in seed_dfs[1:]:
            merged = merged.merge(sdf, on=group_col, how="outer")

        value_cols = [c for c in merged.columns if c.startswith("value_")]
        n_seeds = len(value_cols)

        for _, row in merged.iterrows():
            vals = [row[c] for c in value_cols if pd.notna(row[c])]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

            if len(arr) >= 2:
                se = std / np.sqrt(len(arr))
                t_val = sp_stats.t.ppf((1 + confidence) / 2, df=len(arr) - 1)
                ci_half = t_val * se
            else:
                ci_half = 0.0

            rows.append({
                "method": method,
                group_col: row[group_col],
                "mean": mean,
                "std": std,
                "ci_lo": mean - ci_half,
                "ci_hi": mean + ci_half,
                "n_seeds": len(arr),
            })

    return pd.DataFrame(rows)


def aggregate_client_comm(
    runs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Compute cumulative communication cost per method per round.

    Returns DataFrame with: method, round, cum_mb_mean, cum_mb_std, n_seeds
    """
    rows = []
    for method, seeds in runs.items():
        seed_cums: list[pd.Series] = []
        for seed, data in seeds.items():
            if "rounds_client" not in data:
                continue
            rc = data["rounds_client"]
            per_round = rc.groupby("round").agg(
                total_bytes=("bytes_up", "sum")
            ).reset_index()
            per_round["total_bytes"] += (
                rc.groupby("round")["bytes_down"].sum().values
            )
            per_round["cum_mb"] = per_round["total_bytes"].cumsum() / 1e6
            seed_cums.append(per_round.set_index("round")["cum_mb"])

        if not seed_cums:
            continue

        combined = pd.concat(seed_cums, axis=1)
        for rnd in combined.index:
            vals = combined.loc[rnd].dropna().values
            rows.append({
                "method": method,
                "round": rnd,
                "cum_mb_mean": float(np.mean(vals)),
                "cum_mb_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
            })

    return pd.DataFrame(rows)
