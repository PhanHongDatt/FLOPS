"""Generate synthetic demo data matching the FL log data contract."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_METHODS = ["FLOPS", "FedAvg", "FedProx"]
_DEFAULT_SEEDS = ["42", "43", "44"]
_N_CLIENTS = 10
_N_ROUNDS = 50
_N_CLASSES = 10


def _make_rounds_global(
    method: str, seed: int, n_rounds: int = _N_ROUNDS
) -> pd.DataFrame:
    """Generate synthetic global metrics."""
    rng = np.random.default_rng(seed)
    base_acc = {"FLOPS": 0.82, "FedAvg": 0.75, "FedProx": 0.77}.get(method, 0.73)
    base_loss = {"FLOPS": 0.3, "FedAvg": 0.5, "FedProx": 0.45}.get(method, 0.5)

    rounds = np.arange(1, n_rounds + 1)
    # Logistic-like convergence curve
    t = rounds / n_rounds
    acc = base_acc * (1 - np.exp(-4 * t)) + rng.normal(0, 0.01, n_rounds)
    acc = np.clip(acc, 0, 1)
    loss = base_loss * np.exp(-3 * t) + rng.normal(0, 0.01, n_rounds)
    loss = np.clip(loss, 0.01, 5.0)

    return pd.DataFrame({
        "round": rounds,
        "global_acc": acc,
        "global_loss": loss,
        "eval_split": "test",
    })


def _make_rounds_client(
    method: str, seed: int, n_rounds: int = _N_ROUNDS,
    n_clients: int = _N_CLIENTS,
) -> pd.DataFrame:
    """Generate synthetic per-client per-round data."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    for r in range(1, n_rounds + 1):
        # Each round, fraction of clients participate
        participating = rng.choice(n_clients, size=max(3, n_clients // 2), replace=False)
        for c in range(n_clients):
            part = c in participating
            rows.append({
                "round": r,
                "client_id": f"client_{c}",
                "participated": part,
                "num_samples": int(rng.integers(100, 1000)),
                "local_loss": float(rng.uniform(0.1, 1.5)) if part else 0.0,
                "local_metric": float(rng.uniform(0.4, 0.9)) if part else 0.0,
                "train_time_s": float(rng.uniform(5.0, 60.0)) if part else 0.0,
                "bytes_up": int(rng.integers(500_000, 5_000_000)) if part else 0,
                "bytes_down": int(rng.integers(500_000, 5_000_000)) if part else 0,
            })
    return pd.DataFrame(rows)


def _make_partition(
    seed: int, n_clients: int = _N_CLIENTS, n_classes: int = _N_CLASSES,
) -> pd.DataFrame:
    """Generate Non-IID partition data (Dirichlet-like)."""
    rng = np.random.default_rng(seed + 2)
    rows = []
    for c in range(n_clients):
        # Dirichlet-like: some classes get 0
        probs = rng.dirichlet(np.ones(n_classes) * 0.3)
        total = int(rng.integers(200, 2000))
        counts = (probs * total).astype(int)
        for cls in range(n_classes):
            rows.append({
                "client_id": f"client_{c}",
                "class_id": cls,
                "count": int(counts[cls]),
            })
    return pd.DataFrame(rows)


def _make_per_class(
    method: str, seed: int, n_rounds: int = _N_ROUNDS,
    n_classes: int = _N_CLASSES,
) -> pd.DataFrame:
    """Generate per-class metrics."""
    rng = np.random.default_rng(seed + 3)
    rows = []
    for r in range(1, n_rounds + 1):
        for cls in range(n_classes):
            base = 0.5 + 0.3 * (r / n_rounds)
            # Make minority classes (8, 9) worse
            if cls >= 8:
                base *= 0.6
            p = float(np.clip(base + rng.normal(0, 0.05), 0, 1))
            rec = float(np.clip(base + rng.normal(0, 0.05), 0, 1))
            f1 = 2 * p * rec / (p + rec + 1e-8)
            rows.append({
                "round": r,
                "class_id": cls,
                "precision": p,
                "recall": rec,
                "f1": float(f1),
                "support": int(rng.integers(50, 500)),
            })
    return pd.DataFrame(rows)


def _make_system(
    seed: int, n_rounds: int = _N_ROUNDS,
) -> pd.DataFrame:
    """Generate system metrics."""
    rng = np.random.default_rng(seed + 4)
    rounds = np.arange(1, n_rounds + 1)
    return pd.DataFrame({
        "round": rounds,
        "round_time_s": rng.uniform(10.0, 120.0, n_rounds),
        "stragglers": rng.integers(0, 3, n_rounds),
        "cpu_pct": rng.uniform(40.0, 95.0, n_rounds),
        "mem_mb": rng.uniform(1000.0, 8000.0, n_rounds),
    })


def _make_config(
    method: str, seed: str, n_clients: int = _N_CLIENTS,
    n_rounds: int = _N_ROUNDS,
) -> dict[str, Any]:
    """Generate a config.yaml dict."""
    return {
        "method": method,
        "seed": int(seed),
        "n_clients": n_clients,
        "fraction": 0.5,
        "local_epochs": 5,
        "lr": 0.01,
        "n_rounds": n_rounds,
        "git_commit": "demo000000",
    }


def generate_demo_data(
    out_dir: Path,
    methods: list[str] | None = None,
    seeds: list[str] | None = None,
) -> Path:
    """Generate a complete set of demo run data.

    Returns the path to the generated runs directory.
    """
    methods = methods or _DEFAULT_METHODS
    seeds = seeds or _DEFAULT_SEEDS
    runs_dir = Path(out_dir) / "runs"

    for method in methods:
        for seed in seeds:
            seed_int = int(seed)
            run_dir = runs_dir / method / seed
            run_dir.mkdir(parents=True, exist_ok=True)

            # rounds_global.csv
            df = _make_rounds_global(method, seed_int)
            df.to_csv(run_dir / "rounds_global.csv", index=False)

            # rounds_client.csv
            df = _make_rounds_client(method, seed_int)
            df.to_csv(run_dir / "rounds_client.csv", index=False)

            # partition.csv
            df = _make_partition(seed_int)
            df.to_csv(run_dir / "partition.csv", index=False)

            # per_class.csv
            df = _make_per_class(method, seed_int)
            df.to_csv(run_dir / "per_class.csv", index=False)

            # system.csv
            df = _make_system(seed_int)
            df.to_csv(run_dir / "system.csv", index=False)

            # config.yaml
            cfg = _make_config(method, seed)
            with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(cfg, f)

            logger.info("Generated demo data: %s/%s", method, seed)

    return runs_dir
