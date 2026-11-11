"""Global evaluation of one saved round checkpoint, with the server's evaluator (src/federated/server.py).

For a run whose round eval was cut (e.g. s29: training reached round 20, the round-20 eval hit the arm timeout).
Same function, val split, image size, conf and NMS IoU as the run's own centralized eval; the result row has the
round_metrics.csv columns and is written as JSON.

  python scripts/eval_global_checkpoint.py --run <run dir> --round 20 --weights coco:yolo11m.pt+cbam \
      --data-yaml <yolo_root>/data.yaml --out flops_export/s29_round20.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-yaml", type=Path, required=True, help="global 4-class data.yaml")
    ap.add_argument("--view-dir", type=Path, default=Path("/kaggle/tmp/eval_view"))
    ap.add_argument("--device", default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from src.data.coco_view import coco_data_yaml
    from src.experiments.checkpoint import load_global_checkpoint
    from src.model.yolo_wrapper import build_model, evaluate, set_parameters

    cfg = yaml.safe_load((args.run / "config.yaml").read_text(encoding="utf-8"))
    ev = cfg["evaluation"]
    img_size = int(cfg["model"]["image_size"])
    data_yaml = coco_data_yaml(args.data_yaml, args.view_dir) if args.weights.startswith("coco:") else args.data_yaml
    model = build_model(args.weights)
    set_parameters(model, load_global_checkpoint(args.run / "checkpoint" / f"global_round_{args.round:03d}.npz"))
    metrics = evaluate(model=model, data_yaml=data_yaml, img_size=img_size, conf=float(ev["conf"]),
                       iou=float(ev["iou"]), device=args.device, workers=2)
    row = {"run": args.run.name, "round": args.round, "img_size": img_size, "conf": ev["conf"], "iou": ev["iou"],
           **{k: float(v) for k, v in metrics.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=1), encoding="utf-8")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
