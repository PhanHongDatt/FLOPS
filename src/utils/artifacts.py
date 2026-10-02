from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ARTIFACTS_ROOT = Path(__file__).parents[2] / "artifacts"

# Required artifact set per Section 21, CLAUDE.md.
# Split into base (every run) + algorithm-specific (only some algorithms
# produce these). Previously bundled together, causing false "missing artifact"
# warnings for every FedAvg / SCAFFOLD / FedNova run since only
# ClassAwareAggregation writes aggregation_trace.yaml.
BASE_REQUIRED: set[str] = {
    "config.yaml",
    "environment.json",
    "partition_manifest.yaml",
    "metrics.csv",
    "per_class_metrics.csv",
    "round_metrics.csv",
    "run.log",
}

# Centralized (single-node) runs don't have a partition or multi-round loop,
# so drop those two files from the required set. Everything else in §21 still
# applies (config, env, metrics, per-class metrics, log).
CENTRALIZED_REQUIRED: set[str] = {
    "config.yaml",
    "environment.json",
    "metrics.csv",
    "per_class_metrics.csv",
    "run.log",
}

ALGO_SPECIFIC_REQUIRED: dict[str, set[str]] = {
    "ClassAwareAgg": {"aggregation_trace.yaml"},
    "ClassCountFedAvg": {"aggregation_trace.yaml"},
}

# Backwards-compatible union for any external reader that imports the constant.
REQUIRED_ARTIFACTS: set[str] = BASE_REQUIRED | {"aggregation_trace.yaml"}


def make_run_dir(run_id: str) -> Path:
    run_dir = ARTIFACTS_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    return run_dir


def save_yaml(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_environment(run_dir: Path, env: dict[str, Any]) -> None:
    save_json({**env, "saved_at": datetime.utcnow().isoformat()}, run_dir / "environment.json")


def write_run_readme(run_dir: Path, meta: dict[str, Any]) -> None:
    lines = ["# Experiment Run\n"]
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def verify_artifacts(
    run_dir: Path,
    algorithm: str | None = None,
    run_type: str = "federated",
) -> list[str]:
    """Return list of missing required artifacts.

    Args:
        run_dir: run directory to inspect.
        algorithm: FL algorithm name — enables algorithm-specific artefact
            requirements (e.g. ClassAwareAgg → aggregation_trace.yaml).
            When None, only base artefacts for the run_type are enforced.
        run_type: "federated" (default) or "centralized". Centralized runs
            skip partition_manifest.yaml + round_metrics.csv.
    """
    if run_type == "centralized":
        required = set(CENTRALIZED_REQUIRED)
    else:
        required = set(BASE_REQUIRED)
    if algorithm and algorithm in ALGO_SPECIFIC_REQUIRED:
        required |= ALGO_SPECIFIC_REQUIRED[algorithm]
    present = {p.name for p in run_dir.rglob("*") if p.is_file()}
    return sorted(required - present)
