"""ADR-038 ceiling test 1a: retrain only Detect.cv3 of a round checkpoint on pooled labelled data.

  python scripts/retrain_cls_head.py --run <run dir> --round 20 --weights coco:yolo11m.pt \
      --data-yaml <central partition>/data_C0.yaml --out <new run dir> [--device 0]

The recipe is fixed by ADR-038 (defaults below); the output folder holds config.yaml, environment.json and
checkpoint/global_round_XXX.npz, so eval_global_checkpoint.py and eval_size_split.py read it like any run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EPOCHS = 2
LR0 = 0.002          # 0.2 x the clients' lr0 0.01; Ultralytics linear decay to lrf 0.01 within the 2 epochs
RFS_T = 1.0          # as P4t1
BATCH = 8            # as the clients (nbs default: same effective batch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--round", type=int, default=20)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-yaml", type=Path, required=True, help="4-class data yaml of the pooled training set")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="ADR-038 fixes 2; smoke tests only change it")
    ap.add_argument("--fraction", type=float, default=1.0, help="smoke only: Ultralytics 'fraction' of the data")
    args = ap.parse_args()

    from src.data.coco_view import coco_data_yaml
    from src.experiments.checkpoint import CHECKPOINT_TEMPLATE, load_global_checkpoint, save_global_checkpoint
    from src.model.head_retrain import cls_branch_freeze_list, merge_cls_branch
    from src.model.yolo_wrapper import build_model, get_parameters, set_parameters, train_one_round
    from src.preservation.rfs import rfs_data_yaml

    work = args.out / "_work"
    model = build_model(args.weights)
    keys = list(model.model.state_dict())
    before = load_global_checkpoint(args.run / "checkpoint" / CHECKPOINT_TEMPLATE.format(round=args.round))
    set_parameters(model, before)
    data_yaml = coco_data_yaml(args.data_yaml, work / "coco_view") if args.weights.startswith("coco:") else args.data_yaml
    data_yaml = rfs_data_yaml(data_yaml, work / "rfs", RFS_T, seed=0)
    extra = {"freeze": cls_branch_freeze_list(len(model.model.model))}
    if args.fraction < 1.0:
        extra["fraction"] = args.fraction
    metrics = train_one_round(model=model, data_yaml=data_yaml, epochs=args.epochs, batch=BATCH, img_size=args.imgsz,
                              lr0=LR0, device=args.device, project=work, name="retrain", seed=0,
                              extra_overrides=extra)
    after, changed = merge_cls_branch(keys, before, get_parameters(model))
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.run / "config.yaml", args.out / "config.yaml")
    env_src = args.run / "environment.json"
    env = json.loads(env_src.read_text(encoding="utf-8")) if env_src.exists() else {}
    env.update({"exp_id": args.out.name, "cls_head_retrain": {
        "source": args.run.name, "round": args.round, "epochs": args.epochs, "lr0": LR0, "rfs_t": RFS_T,
        "batch": BATCH, "data_yaml": str(args.data_yaml), "fraction": args.fraction,
        "n_changed_tensors": len(changed), "train_metrics": {k: float(v) for k, v in metrics.items()}}})
    (args.out / "environment.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    save_global_checkpoint(args.out / "checkpoint", args.round, after)
    shutil.rmtree(work, ignore_errors=True)
    print(f"{args.out.name}: cv3 retrained from {args.run.name}; {len(changed)} tensors changed, all in cv3")


if __name__ == "__main__":
    main()
