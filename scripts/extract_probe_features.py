"""Features at GT box centres for the ADR-036 probe: neck output, cv3[l][1] and the class readout, every level.

  python scripts/extract_probe_features.py --data-yaml <root4>/data.yaml --weights coco:yolo11m.pt \
      --ckpt <run>/checkpoint/global_round_020.npz --tag s26 --out <dir> [--device 0] [--limit 20]

``--ckpt init`` keeps the COCO weights (round 0). Images are letterboxed like the evaluator's rect batches
(long side ``--imgsz``, short side padded to a multiple of 32, centred). Kept boxes: every bus and truck box and a
model-independent hash sample of car boxes (``src.evaluation.probe.keep_box``), so every checkpoint sees the same
boxes in the same order. Output ``<out>/<tag>.npz``: image, cls, half, level, box (input px), and per level l
``neck_l``, ``cv3_1_l`` (float16) and ``logit_l`` [N, 3] (car, bus, truck rows of the head, float32).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import yaml

from src.evaluation.error_analysis import yolo_labels_to_xyxy
from src.evaluation.probe import (
    PROBE_CLASSES,
    STRIDES,
    centre_cells,
    image_half,
    keep_box,
    primary_level,
)


def letterbox(img: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, int, int]:
    """Resize the long side to ``imgsz``, pad the short side to a multiple of 32 (centred); returns img, r, left, top."""
    import cv2

    h, w = img.shape[:2]
    r = imgsz / max(h, w)
    nw, nh = round(w * r), round(h * r)
    out_w, out_h = int(np.ceil(nw / 32) * 32), int(np.ceil(nh / 32) * 32)
    left, top = round((out_w - nw) / 2 - 0.1), round((out_h - nh) / 2 - 0.1)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((out_h, out_w, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


class Taps:
    """Forward hooks on Detect: its inputs (neck), cv3[l][1] and cv3[l][2] outputs."""

    def __init__(self, detect):
        self.neck, self.cv31, self.logit = [None] * 3, [None] * 3, [None] * 3
        self.handles = [detect.register_forward_pre_hook(self._pre)]
        for lvl in range(len(STRIDES)):
            self.handles.append(detect.cv3[lvl][1].register_forward_hook(self._store(self.cv31, lvl)))
            self.handles.append(detect.cv3[lvl][2].register_forward_hook(self._store(self.logit, lvl)))

    def _pre(self, _module, args):
        for lvl, t in enumerate(args[0]):
            self.neck[lvl] = t.detach()

    @staticmethod
    def _store(slot, lvl):
        def hook(_module, _inp, out):
            slot[lvl] = out.detach()
        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def boxes_for(label_file: Path, w: int, h: int, image: str) -> tuple[np.ndarray, np.ndarray]:
    lines = label_file.read_text(encoding="utf-8").splitlines() if label_file.exists() else []
    xyxy, cls = yolo_labels_to_xyxy(lines, w, h)
    keep = np.array([keep_box(image, i, int(c)) for i, c in enumerate(cls)], dtype=bool)
    return xyxy[keep], cls[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", type=Path, required=True)
    ap.add_argument("--weights", required=True, help="coco:yolo11m.pt / coco:yolo11s.pt")
    ap.add_argument("--ckpt", required=True, help="global_round_XXX.npz, or 'init' for the COCO weights")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke)")
    args = ap.parse_args()

    import cv2
    import torch

    from src.experiments.checkpoint import load_global_checkpoint
    from src.model.yolo_wrapper import build_model, set_parameters, target_class_ids

    data = yaml.safe_load(args.data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path") or args.data_yaml.parent)
    val_dir = root / data["val"]
    label_dir = root / str(data["val"]).replace("images", "labels", 1)
    model = build_model(args.weights)
    if args.ckpt != "init":
        set_parameters(model, load_global_checkpoint(Path(args.ckpt)))
    ids = target_class_ids(model)
    rows = [ids[c] for c in PROBE_CLASSES]
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() and torch.cuda.is_available() else "cpu")
    net = model.model.to(device).float().eval()
    taps = Taps(net.model[-1])

    images = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.limit:
        images = images[:args.limit]
    rec: dict[str, list] = {k: [] for k in ("image", "cls", "half", "level", "box")}
    for lvl in range(len(STRIDES)):
        for k in ("neck", "cv3_1", "logit"):
            rec[f"{k}_{lvl}"] = []
    with torch.no_grad():
        for n, path in enumerate(images):
            img = cv2.imread(str(path))
            h, w = img.shape[:2]
            xyxy, cls = boxes_for(label_dir / f"{path.stem}.txt", w, h, path.name)
            if not len(cls):
                continue
            lb, r, left, top = letterbox(img, args.imgsz)
            x = torch.from_numpy(lb[:, :, ::-1].transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0).to(device)
            net(x)
            box_in = xyxy * r + np.array([left, top, left, top])
            cells = centre_cells(box_in, [tuple(t.shape[2:]) for t in taps.neck])
            rec["image"] += [path.name] * len(cls)
            rec["cls"].append(cls)
            rec["half"] += [image_half(path.name)] * len(cls)
            rec["level"].append(primary_level(box_in))
            rec["box"].append(box_in.astype(np.float32))
            for lvl in range(len(STRIDES)):
                yy, xx = torch.from_numpy(cells[:, lvl, 0]), torch.from_numpy(cells[:, lvl, 1])
                rec[f"neck_{lvl}"].append(taps.neck[lvl][0][:, yy, xx].T.half().cpu().numpy())
                rec[f"cv3_1_{lvl}"].append(taps.cv31[lvl][0][:, yy, xx].T.half().cpu().numpy())
                rec[f"logit_{lvl}"].append(taps.logit[lvl][0][rows][:, yy, xx].T.float().cpu().numpy())
            if n % 500 == 0:
                print(f"{args.tag}: {n}/{len(images)} images, {len(rec['image'])} boxes", flush=True)
    taps.remove()
    out = {k: (np.array(v) if k in ("image", "half") else np.concatenate(v)) for k, v in rec.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"{args.tag}.npz", **out)
    counts = np.bincount(out["cls"], minlength=3)
    print(f"{args.tag}: {len(out['cls'])} boxes (car {counts[0]}, bus {counts[1]}, truck {counts[2]}) "
          f"-> {args.out / f'{args.tag}.npz'}")


if __name__ == "__main__":
    main()
