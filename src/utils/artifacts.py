from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ARTIFACTS_ROOT = Path(__file__).parents[2] / "artifacts"

# Required artifact set per Section 21, CLAUDE.md
REQUIRED_ARTIFACTS = {
    "config.yaml",
    "environment.json",
    "partition_manifest.yaml",
    "metrics.csv",
    "per_class_metrics.csv",
    "round_metrics.csv",
    "aggregation_trace.yaml",
    "run.log",
}


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


def verify_artifacts(run_dir: Path) -> list[str]:
    """Return list of missing required artifacts."""
    present = {p.name for p in run_dir.rglob("*") if p.is_file()}
    return sorted(REQUIRED_ARTIFACTS - present)
