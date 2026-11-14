"""Make a run folder whose round checkpoint is ``recipient`` with one class row of cv3[l][2] taken from ``donor``.

  python scripts/transplant_rows.py --recipient <run dir> --donor <run dir> --round 20 \
      --weights coco:yolo11m.pt --cls bus --out <new run dir>

The new folder holds config.yaml, environment.json (exp_id names the transplant) and
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

import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipient", type=Path, required=True)
    ap.add_argument("--donor", type=Path, required=True)
    ap.add_argument("--round", type=int, default=20)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cls", default="bus")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from src.experiments.checkpoint import CHECKPOINT_TEMPLATE, load_global_checkpoint, save_global_checkpoint
    from src.model.row_transplant import final_cls_conv_keys, transplant_row
    from src.model.yolo_wrapper import build_model, target_class_ids

    model = build_model(args.weights)
    keys = list(model.model.state_dict())
    class_id = target_class_ids(model)[args.cls]
    name = CHECKPOINT_TEMPLATE.format(round=args.round)
    rec = load_global_checkpoint(args.recipient / "checkpoint" / name)
    don = load_global_checkpoint(args.donor / "checkpoint" / name)
    out = transplant_row(keys, rec, don, class_id)
    changed = [k for k, a, b in zip(keys, rec, out) if not np.array_equal(a, b)]
    if not set(changed) <= set(final_cls_conv_keys(keys)):
        raise SystemExit(f"unexpected changes: {changed}")
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.recipient / "config.yaml", args.out / "config.yaml")
    env_src = args.recipient / "environment.json"
    env = json.loads(env_src.read_text(encoding="utf-8")) if env_src.exists() else {}
    env.update({"exp_id": args.out.name, "transplant": {"recipient": args.recipient.name, "donor": args.donor.name,
                                                         "class": args.cls, "class_id": class_id,
                                                         "changed": changed}})
    (args.out / "environment.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    save_global_checkpoint(args.out / "checkpoint", args.round, out)
    print(f"{args.out.name}: {args.cls} row (id {class_id}) from {args.donor.name}; changed {changed}")


if __name__ == "__main__":
    main()
