"""Train the teacher T on the server's labelled 1,000-image sample (ADR-015).

Centralized multi-epoch training with the G2 schedule (Ultralytics warm-up and mosaic
closing, ADR-009) from the same COCO-initialised nc=4 model the FL runs start from. No
model selection on the val set: the final fp32 EMA is kept. Writes

  <run_dir>/checkpoint/teacher.npz     get_parameters order (= global_round_*.npz format)
  <run_dir>/round_metrics.csv          one row, round 0, same evaluator as the FL runs
  <run_dir>/metrics.csv, per_class_metrics.csv

Usage:
  python scripts/train_server_teacher.py --data-yaml .../server_sample_1k_seed42/data_S.yaml \\
      --global-data-yaml .../bdd100k_yolo/data.yaml --exp-config configs/experiments/convergence.yaml \\
      --epochs 50 --seed 42 --run-class feasibility
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from src.evaluation.metrics import flatten_per_class_metrics, save_per_class_metrics, save_summary_metrics  # noqa: E402
from src.experiments.runner import finalize_run, prepare_run  # noqa: E402
from src.utils.config import build_centralized_train_kwargs, load_experiment_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", type=Path, required=True, help="data_S.yaml of the server sample")
    ap.add_argument("--global-data-yaml", type=Path, required=True)
    ap.add_argument("--exp-config", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-class", choices=["smoke", "feasibility", "main"], default="feasibility")
    ap.add_argument("--partition-id", default="server_sample_1k_seed42")
    args = ap.parse_args()

    from src.federated.server import _append_round_row
    from src.model.yolo_wrapper import build_model, evaluate, get_parameters, train_one_round

    config = load_experiment_config(args.exp_config)
    config["train"]["epochs"] = args.epochs
    run_dir, _ = prepare_run(exp_id="T-teacher", config_path=args.exp_config, run_class=args.run_class,
                             seed=args.seed, partition_id=args.partition_id)
    model = build_model(config["model"]["weights"])
    train_one_round(model=model, data_yaml=args.data_yaml, project=run_dir / "train", name="teacher",
                    **build_centralized_train_kwargs(config, seed=args.seed))
    ckpt = run_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt / "teacher.npz", *get_parameters(model))

    metrics = evaluate(model=model, data_yaml=args.global_data_yaml, img_size=config["model"]["image_size"],
                       conf=config["evaluation"]["conf"], iou=config["evaluation"]["iou"],
                       device=config["train"]["device"])
    _append_round_row(run_dir / "round_metrics.csv", {"round": 0, **metrics})
    save_summary_metrics(metrics, run_dir / "metrics.csv")
    save_per_class_metrics(flatten_per_class_metrics(metrics), run_dir / "per_class_metrics.csv")
    finalize_run(run_dir, run_type="centralized")
    logger.info("Teacher saved to %s; val metrics %s", ckpt / "teacher.npz", metrics)


if __name__ == "__main__":
    main()
