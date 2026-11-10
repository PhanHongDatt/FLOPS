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
from src.mlops.model_card import build_model_card, write_eval_protocol
from src.mlops.tracking import default_tracking_uri, start_tracking
from src.model.yolo_wrapper import build_model, evaluate, get_parameters, train_one_round
from src.utils.config import build_centralized_train_kwargs, load_experiment_config
from src.utils.logger import get_logger
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
    ap.add_argument("--mlflow-uri", default=default_tracking_uri(),
                    help="'none' disables tracking (ADR-031)")
    ap.add_argument("--mlflow-experiment", default="G2-centralized")
    ap.add_argument("--mlflow-required", action="store_true")
    ap.add_argument("--register-model", action="store_true",
                    help="register the final model in the MLflow registry (alias 'candidate')")
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

    write_eval_protocol(
        run_dir, model_weights=str(config["model"]["weights"]),
        image_size=int(config["model"]["image_size"]), eval_conf=float(config["evaluation"]["conf"]),
        eval_iou=float(config["evaluation"]["iou"]), eval_data_yaml=args.data_yaml,
    )
    tracker = start_tracking(run_dir, args.mlflow_experiment, args.mlflow_uri, run_name=f"seed_{args.seed}",
                             tags={"seed": args.seed, "run_class": args.run_class, "exp_id": "G2-centralized",
                                   "run_dir": run_dir.name},
                             required=args.mlflow_required)
    tracker.log_params({"effective": config})
    tracker.log_params({
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

    # seed + Ultralytics schedule defaults (warm-up, close_mosaic): ADR-009
    train_metrics = train_one_round(
        model=model,
        data_yaml=args.data_yaml,
        project=run_dir / "checkpoint",
        name="train",
        **build_centralized_train_kwargs(config, seed=args.seed),
    )
    tracker.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

    eval_metrics = evaluate(
        model=model,
        data_yaml=args.data_yaml,
        img_size=config["model"]["image_size"],
        conf=config["evaluation"]["conf"],
        iou=config["evaluation"]["iou"],
        device=config["train"]["device"],
    )
    tracker.log_metrics({f"eval/{k}": v for k, v in eval_metrics.items()})

    save_summary_metrics(eval_metrics, run_dir / "metrics.csv")
    save_per_class_metrics(
        flatten_per_class_metrics(eval_metrics),
        run_dir / "per_class_metrics.csv",
    )
    save_yaml({**train_metrics, **eval_metrics}, run_dir / "final_metrics.yaml")
    # ADR-031: same package format as FL runs (ordered state_dict of the evaluated model)
    np.savez(run_dir / "checkpoint" / "final_params.npz", *get_parameters(model))

    finalize_run(run_dir, run_type="centralized")
    tracker.log_run_artifacts(run_dir)
    if args.register_model:
        if tracker.enabled:
            from src.mlops.registry import load_policy, register_run
            version = register_run(tracker, run_dir, build_model_card(run_dir), load_policy())
            logger.info("Registered model version %s (alias candidate)", version)
        else:
            logger.warning("--register-model: MLflow tracking is disabled, model NOT registered")
    tracker.end()
    logger.info("Centralized run complete. Metrics: %s", eval_metrics)


if __name__ == "__main__":
    main()
