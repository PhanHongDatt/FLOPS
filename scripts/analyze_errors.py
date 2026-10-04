"""D1 — decompose target-class FN / FP of finished runs (ADR-014).

For every run found under the given roots (an ``artifacts/runs`` tree, or a
``flops_results.zip`` from an earlier Kaggle session), load the global checkpoint of
``--round``, predict the global val set at conf 0.001 and classify every target GT
and every confident target prediction (src/evaluation/error_analysis.py).

Usage:
  python scripts/analyze_errors.py --data-yaml /kaggle/tmp/data/bdd100k_yolo/data.yaml \\
      --roots /kaggle/input --out flops_export/d1_errors.csv --targets bus truck
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from src.data.bdd100k import TARGET_CLASSES  # noqa: E402
from src.evaluation.error_analysis import decompose_image, yolo_labels_to_xyxy  # noqa: E402

CKPT = "checkpoint/global_round_{r:03d}.npz"


def find_runs(roots: list[Path], rnd: int, extract_to: Path) -> list[tuple[dict, Path]]:
    """(environment.json, checkpoint path) for every run that has the round's checkpoint."""
    found: dict[str, tuple[dict, Path]] = {}
    ckpt = CKPT.format(r=rnd)
    for root in roots:
        for env_path in sorted(Path(root).glob("**/environment.json")):
            run = env_path.parent
            if (run / ckpt).exists():
                found.setdefault(run.name, (json.loads(env_path.read_text(encoding="utf-8")), run / ckpt))
        for k, archive in enumerate(sorted(Path(root).glob("**/flops_results.zip"))):
            with zipfile.ZipFile(archive) as z:
                names = set(z.namelist())
                for n in sorted(names):
                    if not (n.startswith("artifacts/runs/") and n.endswith("/environment.json")):
                        continue
                    run_prefix = n[: -len("environment.json")]
                    member = run_prefix + ckpt
                    run_name = run_prefix.rstrip("/").split("/")[-1]
                    if member not in names or run_name in found:
                        continue
                    dest = extract_to / f"zip{k}"
                    z.extract(member, dest)
                    found[run_name] = (json.loads(z.read(n)), dest / member)
    return sorted(found.values(), key=lambda x: (str(x[0].get("partition_id")), str(x[0].get("exp_id")),
                                                 int(x[0].get("seed", 0))))


def analyze_run(model, val_dir: Path, label_dir: Path, targets: list[int], conf_thr: float,
                iou_thr: float, imgsz: int, device, batch: int) -> dict[str, float]:
    gt: dict[int, Counter] = {t: Counter() for t in targets}
    fp: dict[int, Counter] = {t: Counter() for t in targets}
    fn_scores: dict[int, list[float]] = {t: [] for t in targets}
    for res in model.predict(source=str(val_dir), conf=0.001, iou=0.7, imgsz=imgsz, device=device,
                             batch=batch, stream=True, verbose=False):
        h, w = res.orig_shape
        lf = label_dir / (Path(res.path).stem + ".txt")
        lines = lf.read_text(encoding="utf-8").splitlines() if lf.exists() else []
        gb, gc = yolo_labels_to_xyxy(lines, w, h)
        b = res.boxes
        pb, pc, pf = b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy()
        for t in targets:
            g, f, s = decompose_image(gb, gc, pb, pc, pf, t, TARGET_CLASSES, conf_thr, iou_thr)
            gt[t].update(g)
            fp[t].update(f)
            fn_scores[t].extend(s)
    row: dict[str, float] = {}
    for t in targets:
        name = TARGET_CLASSES[t]
        n_gt = sum(gt[t].values())
        row[f"{name}_gt"] = n_gt
        for key in ["tp", "low_score", "missed"] + [f"confused_{c}" for c in TARGET_CLASSES if c != name]:
            row[f"{name}_{key}"] = gt[t].get(key, 0)
        row[f"{name}_fp"] = sum(fp[t].values())
        for key in ["background", "loc"] + [f"on_{c}" for c in TARGET_CLASSES if c != name]:
            row[f"{name}_fp_{key}"] = fp[t].get(key, 0)
        s = fn_scores[t]
        row[f"{name}_fn_score_median"] = statistics.median(s) if s else 0.0
        row[f"{name}_fn_score_ge_0.1"] = sum(v >= 0.1 for v in s)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", type=Path, required=True, help="global YOLO data.yaml (val split)")
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--round", type=int, default=30)
    ap.add_argument("--targets", nargs="+", default=["bus", "truck"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf-thr", type=float, default=0.25)   # FP/FN threshold of ADR-009
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=0)
    ap.add_argument("--extract-to", type=Path, default=Path("/kaggle/tmp/d1_ckpts"))
    args = ap.parse_args()

    from src.experiments.checkpoint import load_global_checkpoint
    from src.model.yolo_wrapper import build_model, set_parameters

    data = yaml.safe_load(args.data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path") or args.data_yaml.parent)
    val_dir = root / data["val"]
    label_dir = root / str(data["val"]).replace("images", "labels", 1)   # YOLO layout
    targets = [TARGET_CLASSES.index(t) for t in args.targets]

    runs = find_runs(args.roots, args.round, args.extract_to)
    if not runs:
        raise SystemExit(f"no run with {CKPT.format(r=args.round)} under {args.roots}")
    print(f"{len(runs)} runs; val {val_dir}")
    rows = []
    for env, ckpt in runs:
        model = build_model("yolov8n.pt")
        set_parameters(model, load_global_checkpoint(ckpt))
        row = {"exp_id": env.get("exp_id"), "partition_id": env.get("partition_id"),
               "seed": env.get("seed"), "round": args.round}
        row.update(analyze_run(model, val_dir, label_dir, targets, args.conf_thr, args.iou_thr,
                               args.imgsz, args.device, args.batch))
        print(json.dumps(row))
        rows.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as f:   # rewrite: partial results survive
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
