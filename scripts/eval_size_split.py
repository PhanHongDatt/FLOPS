"""AP50 per COCO size range for finished runs (docs/Solution.md §6, ADR-032).

Usage (Kaggle, runs mounted as kernel sources):
  python scripts/eval_size_split.py --data-yaml <yolo_root>/data.yaml --roots /kaggle/input \
      --rounds 10 20 --weights coco:yolo11m.pt --include m960 --imgsz 960 --out flops_export/size_split.csv

Predictions: conf 0.001, NMS IoU 0.7, max_det 300 — the settings of the global evaluator — so the "all" row
should match the run's AP50 within interpolation/letterbox differences; it is written as a check.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from analyze_errors import coco_to_target_map, find_runs, remap_predictions  # noqa: E402
from src.data.bdd100k import TARGET_CLASSES  # noqa: E402
from src.evaluation.error_analysis import yolo_labels_to_xyxy  # noqa: E402
from src.evaluation.size_split import AREA_RANGES, ap_from_flags, match_flags  # noqa: E402


def multi_label_predictor():
    """DetectionPredictor whose NMS keeps every class above conf for a box (``multi_label=True``).

    [YOLO-DOC] ultralytics 8.3.253: ``DetectionValidator.postprocess`` (the global evaluator) calls
    ``non_max_suppression(..., multi_label=True)``; ``DetectionPredictor.postprocess`` does not, so ``predict()``
    keeps only the top class of a box. A bus scored just below truck then disappears: with plain ``predict()``
    the bus AP50 of S26 round 20 came out 0.536 against the evaluator's 0.622 (s29e)."""
    import functools

    from ultralytics.models.yolo.detect import DetectionPredictor
    from ultralytics.utils import nms

    class MultiLabelPredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            original = nms.non_max_suppression
            nms.non_max_suppression = functools.partial(original, multi_label=True)
            try:
                return super().postprocess(preds, img, orig_imgs, **kwargs)
            finally:
                nms.non_max_suppression = original

    return MultiLabelPredictor


def collect(model, val_dir: Path, label_dir: Path, imgsz: int, device, batch: int, pred_map):
    """Per class: detections [(image, score, xyxy)] and ground truths {image: [N, 4]}, original pixels."""
    dets = {c: [] for c in range(len(TARGET_CLASSES))}
    gts = {c: {} for c in range(len(TARGET_CLASSES))}
    for res in model.predict(source=str(val_dir), conf=0.001, iou=0.7, max_det=300, imgsz=imgsz,
                             device=device, batch=batch, stream=True, verbose=False,
                             predictor=multi_label_predictor()):
        h, w = res.orig_shape
        stem = Path(res.path).stem
        lf = label_dir / f"{stem}.txt"
        gb, gc = yolo_labels_to_xyxy(lf.read_text(encoding="utf-8").splitlines() if lf.exists() else [], w, h)
        b = res.boxes
        pb, pc, pf = remap_predictions(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy(), pred_map)
        for c in range(len(TARGET_CLASSES)):
            gts[c][stem] = gb[gc == c] if len(gb) else np.zeros((0, 4))
            dets[c].extend((stem, float(s), box) for box, s in zip(pb[pc == c], pf[pc == c]))
    return dets, gts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", type=Path, required=True, help="global 4-class YOLO data.yaml (val split)")
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--rounds", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--weights", required=True, help="architecture, e.g. coco:yolo11m.pt or coco:yolo11m.pt+cbam")
    ap.add_argument("--include", default="", help="only runs whose directory name contains this")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=0)
    ap.add_argument("--only-targets", action="store_true", help="NMS over the four target classes only (ADR-034 P1)")
    ap.add_argument("--dump", type=Path, default=None, help="save per-detection TP flags (npz) for bootstrap CIs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extract-to", type=Path, default=Path("/kaggle/tmp/size_ckpts"))
    args = ap.parse_args()

    from src.experiments.checkpoint import load_global_checkpoint
    from src.model.yolo_wrapper import build_model, set_parameters

    data = yaml.safe_load(args.data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path") or args.data_yaml.parent)
    val_dir = root / data["val"]
    label_dir = root / str(data["val"]).replace("images", "labels", 1)
    pred_map = coco_to_target_map() if args.weights.startswith("coco:") else None
    rows: list[dict] = []
    for rnd in args.rounds:
        runs = [(e, c) for e, c in find_runs(args.roots, rnd, args.extract_to / f"r{rnd}")
                if args.include in c.parent.parent.name]
        print(f"round {rnd}: {len(runs)} runs")
        for env, ckpt in runs:
            model = build_model(args.weights)
            set_parameters(model, load_global_checkpoint(ckpt))
            if args.only_targets:
                from src.model.class_filter import keep_only_classes
                keep_only_classes(model, list(pred_map) if pred_map else list(range(len(TARGET_CLASSES))))
            dets, gts = collect(model, val_dir, label_dir, args.imgsz, args.device, args.batch, pred_map)
            images = sorted(gts[0])
            dump: dict[str, np.ndarray] = {"images": np.array(images)}
            for c, name in enumerate(TARGET_CLASSES):
                for rng_name, rng in AREA_RANGES.items():
                    m = match_flags(dets[c], gts[c], rng, images)
                    ap50, recall = ap_from_flags(m.score, m.flag, int(m.n_gt.sum()))
                    rows.append({"run": ckpt.parent.parent.name, "exp_id": env.get("exp_id"),
                                 "partition_id": env.get("partition_id"), "seed": env.get("seed"), "round": rnd,
                                 "imgsz": args.imgsz, "only_targets": args.only_targets, "class": name,
                                 "size": rng_name, "n_gt": int(m.n_gt.sum()), "AP50": ap50, "recall": recall})
                    for field in ("img", "score", "flag", "n_gt"):
                        dump[f"{name}/{rng_name}/{field}"] = getattr(m, field)
            if args.dump is not None:
                args.dump.mkdir(parents=True, exist_ok=True)
                tag = f"{ckpt.parent.parent.name}_r{rnd}_{args.imgsz}{'_t4' if args.only_targets else ''}"
                np.savez_compressed(args.dump / f"{tag}.npz", **dump)
            print(json.dumps([x for x in rows[-16:] if x["class"] == "motorcycle"]))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w", newline="", encoding="utf-8") as f:   # rewrite: partial results survive
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
