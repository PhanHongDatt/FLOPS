"""F2 — controlled perturbation of class-associated parameters (CLAUDE.md §8 F2).

Loads a trained nc=4 checkpoint, perturbs one parameter group at a time
(class-head rows, or a norm-matched shared control), and evaluates every
variant on the same seeded val subset. Writes metrics.csv,
perturbation_log.yaml and summary.yaml (effect matrix) for the pre-registered
rule in research/feasibility/F2/README.md.

Usage (Kaggle: call through subprocess from notebooks/03_missing_class_G4.py):
  python scripts/run_f2.py \\
    --f2-config configs/feasibility/f2_perturb_bus.yaml \\
    --weights /kaggle/working/g2/best.pt \\
    --eval-data-yaml /kaggle/working/data/bdd100k_yolo/data.yaml
"""
from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data.bdd100k import TARGET_CLASSES
from src.experiments.checkpoint import load_global_checkpoint
from src.experiments.f2 import F2Hooks, build_specs, run_perturbations
from src.experiments.runner import _git_commit, _read_environment_lock
from src.model.yolo_wrapper import (
    build_model,
    confidence_stats,
    evaluate,
    get_parameters,
    set_parameters,
    val_subset_data_yaml,
)
from src.utils.artifacts import save_json, save_yaml
from src.utils.config import load_config, load_experiment_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _load_model(weights: Path, base_weights: str):
    if weights.suffix == ".npz":
        model = build_model(base_weights)
        set_parameters(model, load_global_checkpoint(weights))
        return model
    return build_model(str(weights))


def _hooks(model_factory, val_yaml: Path, images: list[str], ev: dict[str, Any]) -> F2Hooks:
    def state(model) -> dict[str, np.ndarray]:
        names = list(model.model.state_dict().keys())
        return {k: np.array(v, copy=True) for k, v in zip(names, get_parameters(model))}

    def load(model, new_state: dict[str, np.ndarray]) -> None:
        set_parameters(model, [new_state[k] for k in model.model.state_dict()])

    return F2Hooks(
        build=model_factory,
        state=state,
        load=load,
        evaluate=lambda m: evaluate(m, val_yaml, ev["image_size"], ev["conf"], ev["iou"], ev["device"]),
        confidence=lambda m: confidence_stats(m, images, ev["image_size"], ev["conf"], ev["iou"], ev["device"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f2-config", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True, help="trained nc=4 .pt or FL global .npz")
    ap.add_argument("--eval-data-yaml", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("research/feasibility/F2/runs"))
    args = ap.parse_args()

    cfg = load_config(args.f2_config)
    exp = load_experiment_config(Path(cfg["exp_config"]))
    ev = {"image_size": exp["model"]["image_size"], "conf": exp["evaluation"]["conf"],
          "iou": exp["evaluation"]["iou"], "device": exp["train"]["device"]}

    out_dir = args.output_dir / f"{cfg['exp_id']}_{cfg['run_class']}_{args.weights.stem}"
    sub = cfg["val_subset"]
    val_yaml = val_subset_data_yaml(args.eval_data_yaml, out_dir, sub["n_images"], sub["seed"])
    images = Path(yaml.safe_load(val_yaml.read_text(encoding="utf-8"))["val"]).read_text().split()

    specs = build_specs(cfg["perturbations"], TARGET_CLASSES, cfg["target_class"])
    import torch
    import ultralytics
    save_yaml({**cfg, "eval": ev, "n_specs": len(specs)}, out_dir / "config.yaml")
    save_json({**_read_environment_lock(), "python": platform.python_version(),
               "torch": str(torch.__version__), "ultralytics": str(ultralytics.__version__),
               "cuda": torch.version.cuda, "git_commit": _git_commit(),
               "weights": str(args.weights), "eval_data_yaml": str(args.eval_data_yaml),
               "val_subset": sub, "run_class": cfg["run_class"]}, out_dir / "environment.json")

    hooks = _hooks(lambda: _load_model(args.weights, exp["model"]["weights"]), val_yaml, images, ev)
    summary = run_perturbations(specs, TARGET_CLASSES, hooks, out_dir)
    if summary["baseline"].get(f"AP50_{cfg['target_class']}", 0.0) <= 0.0:
        logger.warning("Checkpoint has zero AP50 for %r: perturbation effects on it are "
                       "unmeasurable — use a checkpoint that detects the class.", cfg["target_class"])
    logger.info("F2 done → %s (%d perturbations)", out_dir, len(specs))


if __name__ == "__main__":
    main()
