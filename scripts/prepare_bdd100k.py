"""Prepare BDD100K for YOLOv8 training.

Converts BDD100K annotations (JSON) to YOLO format, filters for the 4
target classes (car, bus, truck, motorcycle), and writes YOLO data.yaml.

Expected input structure (per BDD100K official release):
  <data_root>/
    images/100k/train/*.jpg
    images/100k/val/*.jpg
    labels/det_20/det_train.json    (or bdd100k_labels_images_train.json)
    labels/det_20/det_val.json

Output structure (YOLO-compatible):
  <output_root>/
    images/train/*.jpg   (symlinks or original)
    images/val/*.jpg
    labels/train/*.txt   (YOLO format: cls cx cy w h)
    labels/val/*.txt
    data.yaml            (Ultralytics data config)

Usage:
  python scripts/prepare_bdd100k.py \\
    --data-root /path/to/bdd100k \\
    --output-root data/bdd100k_yolo \\
    --img-w 1280 --img-h 720
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

from src.data.bdd100k import (
    TARGET_CLASSES,
    filter_target_classes,
    load_bdd100k_annotations,
    write_yolo_labels,
)


def _link_or_copy(src: Path, dst: Path, use_symlink: bool = True) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_symlink and os.name != "nt":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _process_split(
    ann_json: Path,
    src_img_dir: Path,
    out_img_dir: Path,
    out_label_dir: Path,
    img_w: int,
    img_h: int,
    use_symlink: bool,
) -> tuple[int, dict[str, int]]:
    if not ann_json.exists():
        raise FileNotFoundError(f"Annotation file missing: {ann_json}")

    all_ann = load_bdd100k_annotations(ann_json)
    kept = filter_target_classes(all_ann)

    write_yolo_labels(kept, out_label_dir, img_w=img_w, img_h=img_h)

    counts = {c: 0 for c in TARGET_CLASSES}
    for frame in kept:
        img_name = frame["name"]
        src = src_img_dir / img_name
        dst = out_img_dir / img_name
        if src.exists():
            _link_or_copy(src, dst, use_symlink)
        for lb in frame["labels"]:
            cat = lb.get("category")
            if cat in counts:
                counts[cat] += 1

    return len(kept), counts


def write_yolo_data_yaml(
    output_root: Path,
    train_img_rel: str = "images/train",
    val_img_rel: str = "images/val",
) -> Path:
    data = {
        "path": str(output_root.resolve()),
        "train": train_img_rel,
        "val": val_img_rel,
        "nc": len(TARGET_CLASSES),
        "names": list(TARGET_CLASSES),
    }
    out = output_root / "data.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True,
                    help="BDD100K root (must contain images/100k and labels)")
    ap.add_argument("--output-root", type=Path, required=True,
                    help="Output root for YOLO-formatted dataset")
    ap.add_argument("--img-w", type=int, default=1280)
    ap.add_argument("--img-h", type=int, default=720)
    ap.add_argument("--no-symlink", action="store_true",
                    help="Copy images instead of symlinking")
    ap.add_argument(
        "--train-ann", type=str, default="labels/det_20/det_train.json",
        help="Path (relative to data-root) to training annotation JSON",
    )
    ap.add_argument(
        "--val-ann", type=str, default="labels/det_20/det_val.json",
        help="Path (relative to data-root) to validation annotation JSON",
    )
    args = ap.parse_args()

    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)

    for split, ann_rel, src_rel in (
        ("train", args.train_ann, "images/100k/train"),
        ("val", args.val_ann, "images/100k/val"),
    ):
        n, counts = _process_split(
            ann_json=args.data_root / ann_rel,
            src_img_dir=args.data_root / src_rel,
            out_img_dir=output / "images" / split,
            out_label_dir=output / "labels" / split,
            img_w=args.img_w,
            img_h=args.img_h,
            use_symlink=not args.no_symlink,
        )
        print(f"[{split}] frames kept: {n} | box counts: {counts}")

    data_yaml = write_yolo_data_yaml(output)
    print(f"Wrote {data_yaml}")


if __name__ == "__main__":
    main()
