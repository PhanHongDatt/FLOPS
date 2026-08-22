"""Centralized YOLOv8n training — G2 baseline.

Runs Ultralytics training on the prepared BDD100K YOLO dataset and
logs to MLflow. Produces the reference baseline that G3+ methods
must be compared against on the same evaluator.

Usage:
  python scripts/train_centralized.py \\
    --data-yaml data/bdd100k_yolo/data.yaml \\
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

from src.experiments.runner import prepare_run, finalize_run
from src.model.yolo_wrapper import build_model, evaluate, train_one_round
from src.utils.config import load_experiment_config
from src.utils.logger import get_logger, setup_mlflow, log_metrics, log_params, end_run
from src.utils.artifacts import save_yaml
from src.evaluation.metrics import (
    flatten_per_class_metrics,
    save_per_class_metrics,
    save_summary_metrics,
)

logger = get_logger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", type=Path, required=True)
    ap.add_argument("--exp-config", type=Path, default=None,
                    help="Experiment override config (e.g. configs/experiments/smoke.yaml)")
    ap.add_argument("--run-class", choices=["smoke", "feasibility", "main"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow-uri", default="http://mlflow:5000")
    ap.add_argument("--mlflow-experiment", default="G2-centralized")
    args = ap.parse_args()

    _set_seed(args.seed)

    # Resolve config: base + experiment override
    if args.exp_config is None:
        args.exp_config = Path(__file__).parents[1] / "configs" / "experiments" / f"{args.run_class}.yaml"
    config = load_experiment_config(args.exp_config)

    run_dir, _ = prepare_run(
        exp_id="G2-centralized",
        config_path=args.exp_config,
        run_class=args.run_class,
        seed=args.seed,
        partition_id="centralized",
    )

    run_id = setup_mlflow(args.mlflow_experiment, args.mlflow_uri, run_name=f"seed_{args.seed}")
    log_params({
        "seed": args.seed,
        "run_class": args.run_class,
        "model": config["model"]["name"],
        "image_size": config["model"]["image_size"],
        "epochs": config["train"]["epochs"],
        "batch_size": config["train"]["batch_size"],
        "lr0": config["train"]["lr0"],
        "optimizer": config["train"]["optimizer"],
        "data_yaml": str(args.data_yaml),
    })

    model = build_model(config["model"]["weights"])

    train_metrics = train_one_round(
        model=model,
        data_yaml=args.data_yaml,
        epochs=config["train"]["epochs"],
        batch=config["train"]["batch_size"],
        img_size=config["model"]["image_size"],
        lr0=config["train"]["lr0"],
        device=config["train"]["device"],
        project=run_dir / "checkpoint",
        name="train",
    )
    log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

    eval_metrics = evaluate(
        model=model,
        data_yaml=args.data_yaml,
        img_size=config["model"]["image_size"],
        conf=config["evaluation"]["conf"],
        iou=config["evaluation"]["iou"],
        device=config["train"]["device"],
    )
    log_metrics(eval_metrics)

    save_summary_metrics(eval_metrics, run_dir / "metrics.csv")
    save_per_class_metrics(
        flatten_per_class_metrics(eval_metrics),
        run_dir / "per_class_metrics.csv",
    )
    save_yaml({**train_metrics, **eval_metrics}, run_dir / "final_metrics.yaml")

    end_run()
    finalize_run(run_dir)
    logger.info("Centralized run complete. Metrics: %s", eval_metrics)


if __name__ == "__main__":
    main()
