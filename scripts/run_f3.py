"""F3 — matched missing-class local training (CLAUDE.md §8, plan.md §6.2).

Trains one global checkpoint locally on a matched pair (client without the
target class vs its S1-Control-Matched twin) for every seed and writes
parameter + prediction evidence. Local training uses exactly the FL client
settings (``build_local_train_config``), so Δθ describes an FL client update.

Usage (Kaggle: call through subprocess from notebooks/03_missing_class_G4.py):
  python scripts/run_f3.py \\
    --f3-config configs/feasibility/f3_matched_bus.yaml \\
    --partition-dir /kaggle/working/partitions \\
    --global-weights /kaggle/working/g2/best.pt \\
    --eval-data-yaml /kaggle/working/bdd100k_yolo/data.yaml

``--global-weights`` accepts an Ultralytics ``.pt`` checkpoint (e.g. the G2
centralized best.pt) or an FL global checkpoint ``global_round_XXX.npz``.
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
from src.experiments.f3 import F3Hooks, resolve_side, run_matched_pair, validate_pair
from src.experiments.runner import _git_commit, _read_environment_lock
from src.model.yolo_wrapper import (
    build_model,
    confidence_stats,
    evaluate,
    get_parameters,
    set_parameters,
    train_one_round,
    val_subset_data_yaml,
)
from src.utils.artifacts import save_json, save_yaml
from src.utils.config import (
    build_local_train_config,
    load_config,
    load_experiment_config,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _model_factory(global_weights: Path, base_weights: str):
    """Fresh model holding the global checkpoint, identical on every call."""
    if global_weights.suffix == ".npz":
        params = load_global_checkpoint(global_weights)

        def build():
            model = build_model(base_weights)
            set_parameters(model, params)
            return model
        return build
    return lambda: build_model(str(global_weights))


def _hooks(global_weights: Path, base_weights: str, eval_data_yaml: Path, tc: dict[str, Any],
           conf_images: list[str]) -> F3Hooks:
    def train(model, data_yaml: Path, seed: int, project: Path, name: str) -> None:
        train_one_round(
            model=model, data_yaml=data_yaml, epochs=tc["local_epochs"], batch=tc["batch_size"],
            img_size=tc["image_size"], lr0=tc["lr0"], device=tc["device"], project=project,
            name=name, seed=seed, workers=int(tc["workers"]),
            warmup_epochs=float(tc["warmup_epochs"]), close_mosaic=int(tc["close_mosaic"]),
            run_val=bool(tc["client_run_val"]), deterministic=bool(tc["deterministic"]),
            nbs=tc["nbs"],
        )

    def run_eval(model) -> dict[str, float]:
        metrics = evaluate(model=model, data_yaml=eval_data_yaml, img_size=tc["image_size"],
                           conf=tc["conf"], iou=tc["iou"], device=tc["device"])
        # CLAUDE.md §8 F3 "confidence change": per-class count/mean confidence on a
        # fixed seeded subset (AP above uses the full val set)
        return {**metrics, **confidence_stats(model, conf_images, tc["image_size"],
                                              tc["conf"], tc["iou"], tc["device"])}

    def state(model) -> dict[str, np.ndarray]:
        # copies: get_parameters returns views of live CPU tensors
        names = list(model.model.state_dict().keys())
        return {k: np.array(v, copy=True) for k, v in zip(names, get_parameters(model))}

    return F3Hooks(build=_model_factory(global_weights, base_weights), train=train,
                   evaluate=run_eval, state=state)


def _environment(f3_cfg: dict, args: argparse.Namespace, match_report: dict | None) -> dict:
    import torch
    import ultralytics
    return {
        **_read_environment_lock(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "git_commit": _git_commit(),
        "global_weights": str(args.global_weights),
        "eval_data_yaml": str(args.eval_data_yaml),
        "seeds": f3_cfg["seeds"],
        "run_class": f3_cfg["run_class"],
        "control_matched": None if match_report is None else bool(match_report.get("matched")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f3-config", type=Path, required=True)
    ap.add_argument("--partition-dir", type=Path, required=True,
                    help="--output-dir used by generate_partition.py")
    ap.add_argument("--global-weights", type=Path, required=True)
    ap.add_argument("--eval-data-yaml", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("research/feasibility/F3/runs"))
    args = ap.parse_args()

    f3_cfg = load_config(args.f3_config)
    exp_config = load_experiment_config(Path(f3_cfg["exp_config"]))
    tc = build_local_train_config(exp_config)
    target = f3_cfg["target_class"]

    missing = resolve_side("missing", args.partition_dir / f3_cfg["missing"]["manifest"],
                           f3_cfg["missing"]["client"], target)
    control = resolve_side("control", args.partition_dir / f3_cfg["control"]["manifest"],
                           f3_cfg["control"]["client"], target)
    validate_pair(missing, control)  # before anything is written to out_dir

    report_path = control.data_yaml.parent / "match_report.yaml"
    match_report = yaml.safe_load(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    if not (match_report or {}).get("matched", False):
        logger.warning(
            "Control %s is not confirmed matched (%s). Results may be reported as "
            "observations only, not as a causal Missing-Class effect (CLAUDE.md §10).",
            control.partition_id, report_path,
        )

    run_id = f"{f3_cfg['exp_id']}_{f3_cfg['run_class']}_{missing.partition_id}_{missing.client_id}"
    out_dir = args.output_dir / run_id
    save_yaml({**f3_cfg, "local_train_config": tc}, out_dir / "config.yaml")
    save_json(_environment(f3_cfg, args, match_report), out_dir / "environment.json")
    if match_report is not None:
        save_yaml(match_report, out_dir / "match_report.yaml")

    sub = f3_cfg.get("confidence_subset", {"n_images": 2000, "seed": 0})
    conf_yaml = val_subset_data_yaml(args.eval_data_yaml, out_dir, sub["n_images"], sub["seed"])
    conf_images = Path(yaml.safe_load(conf_yaml.read_text(encoding="utf-8"))["val"]).read_text().split()
    hooks = _hooks(args.global_weights, exp_config["model"]["weights"], args.eval_data_yaml, tc,
                   conf_images)
    summary = run_matched_pair(missing, control, target, TARGET_CLASSES, f3_cfg["seeds"], hooks, out_dir)
    logger.info("F3 done → %s | target_known_before=%s | bias_delta_gap=%s",
                out_dir, summary["target_known_before"], summary["parameter"]["bias_delta_gap"])


if __name__ == "__main__":
    main()
