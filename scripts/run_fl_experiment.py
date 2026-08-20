"""Run federated experiment — G3+ baseline runner.

Executes a Flower simulation with the chosen FL algorithm on a
prepared partition of BDD100K. Records to MLflow + local artifacts.

Usage:
  python scripts/run_fl_experiment.py \\
    --partition configs/partition/s0_iid.yaml \\
    --algorithm FedAvg \\
    --run-class smoke \\
    --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.partitioner import load_partition_manifest
from src.experiments.runner import prepare_run, finalize_run
from src.federated.client import YOLOFlowerClient, make_client_fn
from src.federated.client_variants import FedNovaClient, FedProxClient, ScaffoldClient
from src.federated.server import run_fl_server
from src.utils.config import load_experiment_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLIENT_CLASSES = {
    "FedAvg": YOLOFlowerClient,
    "FedProx": FedProxClient,
    "SCAFFOLD": ScaffoldClient,
    "FedNova": FedNovaClient,
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_client_fn(
    client_cls,
    partition_manifest,
    data_yaml_map,
    weights: str,
    train_config: dict,
    run_dir: Path,
):
    def client_fn(cid: str):
        return client_cls(
            client_id=cid,
            partition_id=int(cid),
            data_yaml=data_yaml_map[cid],
            weights=weights,
            train_config=train_config,
            run_dir=run_dir,
        )
    return client_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", type=Path, required=True,
                    help="Path to partition manifest YAML")
    ap.add_argument("--algorithm", choices=list(CLIENT_CLASSES.keys()), required=True)
    ap.add_argument("--data-yaml-dir", type=Path, required=True,
                    help="Directory containing per-client data_C{i}.yaml files")
    ap.add_argument("--exp-config", type=Path, default=None)
    ap.add_argument("--run-class", choices=["smoke", "feasibility", "main"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow-uri", default="http://mlflow:5000")
    ap.add_argument("--mlflow-experiment", default="G3-federated")
    args = ap.parse_args()

    _set_seed(args.seed)

    if args.exp_config is None:
        args.exp_config = Path(__file__).parents[1] / "configs" / "experiments" / f"{args.run_class}.yaml"
    config = load_experiment_config(args.exp_config)

    manifest = load_partition_manifest(args.partition)
    data_yaml_map = {
        cid: args.data_yaml_dir / f"data_{cid}.yaml"
        for cid in manifest.client_assignments
    }
    for cid, p in data_yaml_map.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing per-client data yaml: {p}")

    run_dir, _ = prepare_run(
        exp_id=f"G3-{args.algorithm}",
        config_path=args.exp_config,
        run_class=args.run_class,
        seed=args.seed,
        partition_id=manifest.partition_id,
    )

    # Save partition manifest into run dir per Section 21
    with (run_dir / "partition_manifest.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(manifest.to_dict(), f, default_flow_style=False, allow_unicode=True)

    train_config = {
        "local_epochs": config["federated"]["local_epochs"],
        "batch_size": config["train"]["batch_size"],
        "image_size": config["model"]["image_size"],
        "lr0": config["train"]["lr0"],
        "conf": config["evaluation"]["conf"],
        "iou": config["evaluation"]["iou"],
        "device": config["train"]["device"],
    }

    client_cls = CLIENT_CLASSES[args.algorithm]
    client_fn = _build_client_fn(
        client_cls, manifest, data_yaml_map,
        weights=config["model"]["weights"],
        train_config=train_config,
        run_dir=run_dir,
    )

    fl_config = {
        "num_rounds": config["federated"]["num_rounds"],
        "num_clients": config["federated"]["num_clients"],
        "fraction_fit": config["federated"]["fraction_fit"],
        "fraction_evaluate": config["federated"]["fraction_evaluate"],
    }

    run_fl_server(
        algorithm=args.algorithm,
        fl_config=fl_config,
        client_fn=client_fn,
        run_dir=run_dir,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_uri=args.mlflow_uri,
    )

    finalize_run(run_dir)
    logger.info("FL run complete: %s / seed=%d", args.algorithm, args.seed)


if __name__ == "__main__":
    main()
