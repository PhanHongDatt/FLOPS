"""N1 (ADR-020) — CCVR-style post-hoc classifier calibration of a finished federated run.

For a global checkpoint: every client computes per-level, per-class feature statistics on its OWN training
images (letterboxed, no augmentation); the server merges them, re-fits only the target rows of the class conv
of each Detect level on virtual features, and the calibrated model is evaluated on the global val set with the
same evaluator as the federated runs. The uncalibrated checkpoint is re-evaluated in the same process.

Usage:
  python scripts/calibrate_classifier.py --checkpoint RUN/checkpoint/global_round_020.npz \\
      --partition configs/partition/s1b5_bus_3k_moto_seed42.yaml --data-yaml-dir DIR \\
      --global-data-yaml DATA.yaml --exp-config configs/experiments/convergence_5c_y11s_20r.yaml \\
      --weights coco:yolo11s.pt --out flops_export/n1/s1b5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from src.preservation.classifier_calibration import calibrate, collect_stats, merge_all  # noqa: E402


def client_batches(data_yaml: Path, imgsz: int, batch: int, workers: int):
    """Val-style (letterbox, no augmentation) batches over a client's training images."""
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_yaml))
    cfg = get_cfg(overrides={"imgsz": imgsz, "rect": False, "cache": False})
    ds = build_yolo_dataset(cfg, data["train"], batch, data, mode="val", rect=False, stride=32)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                                       collate_fn=ds.collate_fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--partition", type=Path, required=True)
    ap.add_argument("--data-yaml-dir", type=Path, required=True)
    ap.add_argument("--global-data-yaml", type=Path, required=True)
    ap.add_argument("--exp-config", type=Path, required=True)
    ap.add_argument("--weights", default="coco:yolo11s.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    # ADR-020, fixed before any run
    ap.add_argument("--n-pos", type=int, default=2000)
    ap.add_argument("--n-bg", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bg-per-image", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from src.data.coco_view import coco_data_yaml
    from src.data.partitioner import load_partition_manifest
    from src.experiments.checkpoint import load_global_checkpoint
    from src.model.yolo_wrapper import build_model, evaluate, get_parameters, set_parameters, target_class_ids
    from src.utils.config import load_experiment_config

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_experiment_config(args.exp_config)
    ev = cfg["evaluation"]
    imgsz = int(ev.get("image_size", cfg["train"].get("image_size", 640)))
    coco = args.weights.startswith("coco:")
    view = args.out / "coco_view"
    gyaml = coco_data_yaml(args.global_data_yaml, view) if coco else args.global_data_yaml
    params = load_global_checkpoint(args.checkpoint)

    def fresh():
        m = build_model(args.weights)
        set_parameters(m, params)
        return m

    def ev_model(m) -> dict[str, float]:
        return evaluate(m, gyaml, imgsz, float(ev["conf"]), float(ev["iou"]), args.device)

    model = fresh()
    det_model = model.model.to(f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device))
    target_ids = sorted(target_class_ids(model).values())
    manifest = load_partition_manifest(args.partition)
    t0 = time.time()
    per_client = []
    for k, cid in enumerate(sorted(manifest.client_assignments)):
        dy = args.data_yaml_dir / f"data_{cid}.yaml"
        dy = coco_data_yaml(dy, view) if coco else dy
        st = collect_stats(det_model, client_batches(dy, imgsz, args.batch, args.workers), target_ids,
                           device=next(det_model.parameters()).device, bg_per_image=args.bg_per_image,
                           seed=args.seed + k)
        np.savez(args.out / f"stats_{cid}.npz", **st.to_npz_dict())      # what the client would send
        per_client.append(st)
        print(cid, {f"{l}_{c}": int(v["n"]) for (l, c), v in sorted(st.sums.items())}, flush=True)
    stats = merge_all(per_client)
    det_model.float().cpu()
    report = calibrate(det_model, stats, target_ids, n_pos=args.n_pos, n_bg=args.n_bg, steps=args.steps,
                       lr=args.lr, seed=args.seed)
    calibrated = get_parameters(model)
    np.savez(args.out / "calibrated.npz", *calibrated)
    t_cal = time.time() - t0

    after = ev_model(model)
    before = ev_model(fresh())
    result = {"checkpoint": str(args.checkpoint), "partition": str(args.partition), "weights": args.weights,
              "target_ids": target_ids, "settings": {k: getattr(args, k) for k in
                                                     ("n_pos", "n_bg", "steps", "lr", "bg_per_image", "seed")},
              "calibration_seconds": round(t_cal, 1), "report": {str(k): v for k, v in report.items()},
              "counts": {f"{l}_{c}": int(v["n"]) for (l, c), v in sorted(stats.sums.items())},
              "before": before, "after": after}
    (args.out / "n1_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    keys = ["mAP50"] + [k for k in after if k.startswith("AP50_")]
    for k in keys:
        print(f"{k:16s} before {before.get(k, float('nan')):.4f}  after {after.get(k, float('nan')):.4f}")


if __name__ == "__main__":
    main()
