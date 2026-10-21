"""Re-evaluate round-20 global checkpoints on the local copy of the global val set, for the report figures.

For each run: one Ultralytics ``val()`` pass (conf 0.001, IoU 0.7, the protocol of every run) gives AP50 bus,
the bus precision-recall curve and the confusion matrix [YOLO-DOC: ultralytics 8.3.253 utils/metrics.py
ap_per_class / ConfusionMatrix: the matrix is indexed [predicted, actual], uses conf > 0.25 and IoU > 0.45, and
background is index nc]. One ``predict()`` pass (conf 0.001) gives, for every ground-truth bus, the highest bus
score among predictions with IoU >= 0.5 (0 if none), and the D1 outcome of every ground-truth bus at score 0.25
(src/evaluation/error_analysis.py).

The local val set must be the run's global val set: 9,905 images, 1,597 bus boxes (checked at start).
Results: one JSON per run under --out; the AP50 bus of the pass is printed next to the registered round-20 value.

Usage:
  python scripts/eval_checkpoints_local.py --runs kaggle/output/s12c_g/artifacts/runs/P4noKD_... [...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.data.bdd100k import TARGET_CLASSES  # noqa: E402
from src.evaluation.error_analysis import box_iou, decompose_image, yolo_labels_to_xyxy  # noqa: E402

N_VAL_IMAGES, N_VAL_BUS = 9905, 1597
BUS = TARGET_CLASSES.index("bus")


def coco_ids() -> dict[str, int]:
    from src.model.yolo_wrapper import COCO_IDS
    return {c: int(COCO_IDS[c]) for c in TARGET_CLASSES}


def check_val_set(data80: Path) -> tuple[Path, Path]:
    data = yaml.safe_load(data80.read_text(encoding="utf-8"))
    root = Path(data["path"])
    img_dir, lab_dir = root / data["val"], root / str(data["val"]).replace("images", "labels", 1)
    images = sorted(img_dir.glob("*.jpg"))
    bus_id = coco_ids()["bus"]
    n_bus = sum(ln.split()[0] == str(bus_id) for f in lab_dir.glob("*.txt")
                for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
    if len(images) != N_VAL_IMAGES or n_bus != N_VAL_BUS:
        sys.exit(f"local val is not the global val set: {len(images)} images, {n_bus} bus (expected "
                 f"{N_VAL_IMAGES}, {N_VAL_BUS})")
    return img_dir, lab_dir


def confusion_5x5(matrix: np.ndarray, ids: dict[str, int]) -> list[list[float]]:
    """Rows = actual (4 targets + background), columns = predicted (4 targets + 'background or other class')."""
    m = np.asarray(matrix, dtype=float)
    bg = m.shape[0] - 1
    t = [ids[c] for c in TARGET_CLASSES]
    rows = []
    for a in t:
        col = m[:, a]
        rows.append([float(col[p]) for p in t] + [float(col.sum() - col[t].sum())])
    rows.append([float(m[p, bg]) for p in t] + [0.0])
    return rows


def val_pass(model, data80: Path, device, batch: int) -> dict:
    ids = coco_ids()
    res = model.val(data=str(data80), imgsz=640, conf=0.001, iou=0.7, device=device, batch=batch,
                    workers=0, plots=False, verbose=False)
    box = res.box
    k = list(box.ap_class_index).index(ids["bus"])
    px = np.asarray(box.px)
    i25 = int(np.argmin(np.abs(px - 0.25)))
    recall_grid = np.linspace(0, 1, 1000)
    return {
        "ap50_bus": float(box.ap50[k]),
        "pr_recall": recall_grid[::10].tolist(),
        "pr_precision": np.asarray(box.prec_values)[k][::10].tolist(),
        "p_at_025": float(np.asarray(box.p_curve)[k][i25]),
        "r_at_025": float(np.asarray(box.r_curve)[k][i25]),
        "confusion_5x5": confusion_5x5(res.confusion_matrix.matrix, ids),
    }


def predict_pass(model, img_dir: Path, lab_dir: Path, device, batch: int) -> dict:
    ids = coco_ids()
    back = {v: TARGET_CLASSES.index(c) for c, v in ids.items()}
    outcomes: Counter = Counter()
    best_scores: list[float] = []
    for r in model.predict(source=str(img_dir), conf=0.001, iou=0.7, imgsz=640, device=device, batch=batch,
                           stream=True, verbose=False):
        h, w = r.orig_shape
        lf = lab_dir / (Path(r.path).stem + ".txt")
        lines = lf.read_text(encoding="utf-8").splitlines() if lf.exists() else []
        gb, gc80 = yolo_labels_to_xyxy(lines, w, h)
        keep_g = np.array([int(c) in back for c in gc80], dtype=bool)
        gb, gc = gb[keep_g], np.array([back[int(c)] for c in gc80[keep_g]], dtype=int)
        b = r.boxes
        pc80 = b.cls.cpu().numpy().astype(int)
        keep = np.array([int(c) in back for c in pc80], dtype=bool)
        pb, pf = b.xyxy.cpu().numpy()[keep], b.conf.cpu().numpy()[keep]
        pc = np.array([back[int(c)] for c in pc80[keep]], dtype=int)
        g_out, _, _ = decompose_image(gb, gc, pb, pc, pf, BUS, TARGET_CLASSES, 0.25, 0.5)
        outcomes.update(g_out)
        iou = box_iou(gb, pb)
        for g in np.flatnonzero(gc == BUS):
            s = pf[(iou[g] >= 0.5) & (pc == BUS)]
            best_scores.append(float(s.max()) if s.size else 0.0)
    return {"bus_outcomes": dict(outcomes), "bus_best_scores": best_scores}


def registered_r20(run: Path) -> float | None:
    rows = {int(r["round"]): r for r in csv.DictReader((run / "round_metrics.csv").open(encoding="utf-8"))}
    return float(rows[20]["AP50_bus"]) if 20 in rows else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--data80", type=Path, default=_REPO_ROOT / "data" / "local_val" / "root80" / "data.yaml")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "research" / "figures" / "data" / "local_eval")
    ap.add_argument("--device", default=0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from src.experiments.checkpoint import load_global_checkpoint
    from src.model.yolo_wrapper import build_model, set_parameters

    img_dir, lab_dir = check_val_set(args.data80)
    args.out.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        dest = args.out / f"{run.name}.json"
        if dest.exists():
            print("skip (done)", run.name)
            continue
        env = json.loads((run / "environment.json").read_text(encoding="utf-8"))
        model = build_model("coco:yolo11s.pt")
        set_parameters(model, load_global_checkpoint(run / "checkpoint" / "global_round_020.npz"))
        out = {"run": run.name, "exp_id": env.get("exp_id"), "seed": env.get("seed"),
               "partition_id": env.get("partition_id"), "round": 20,
               "registered_ap50_bus_r20": registered_r20(run)}
        out.update(predict_pass(model, img_dir, lab_dir, args.device, args.batch))
        out.update(val_pass(model, args.data80, args.device, args.batch))
        assert len(out["bus_best_scores"]) == N_VAL_BUS, len(out["bus_best_scores"])
        dest.write_text(json.dumps(out), encoding="utf-8")
        print(f"{run.name}: AP50 bus {out['ap50_bus']:.4f} (registered r20 {out['registered_ap50_bus_r20']:.4f}), "
              f"outcomes {out['bus_outcomes']}")


if __name__ == "__main__":
    main()
