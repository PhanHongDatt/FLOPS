"""Normalize local BDD100K dataset structure for Kaggle upload.

Converts the messy downloaded structure into the exact layout that
scripts/prepare_bdd100k.py and notebooks/01_smoke_G1_G3.py expect.

INPUT (what you downloaded):
  D:\FLOPS\bdd100k\
    bdd100k\bdd100k\images\100k\
      train\          <- has both .jpg files + trainA/, trainB/ subfolders
        trainA\*.jpg
        trainB\*.jpg
        *.jpg          <- some loose images
      val\*.jpg
    bdd100k_labels_release\bdd100k\labels\
      bdd100k_labels_images_train.json
      bdd100k_labels_images_val.json

OUTPUT (what Kaggle + scripts need):
  D:\FLOPS\bdd100k_kaggle\    <- new folder, original untouched
    images\100k\
      train\*.jpg    <- all ~63k train images flat
      val\*.jpg      <- all 10k val images flat
    labels\det_20\
      det_train.json  <- renamed from bdd100k_labels_images_train.json
      det_val.json    <- renamed from bdd100k_labels_images_val.json

Usage:
  python scripts/normalize_bdd100k_local.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def flatten_images(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for jpg in src_dir.rglob("*.jpg"):
        dst = dst_dir / jpg.name
        if not dst.exists():
            shutil.copy2(jpg, dst)
        copied += 1
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize BDD100K for Kaggle upload")
    ap.add_argument("--src", type=Path, default=Path(r"D:\FLOPS\bdd100k"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\FLOPS\bdd100k_kaggle"))
    args = ap.parse_args()

    src: Path = args.src
    out: Path = args.out

    img_root = src / "bdd100k" / "bdd100k" / "images" / "100k"
    labels_root = src / "bdd100k_labels_release" / "bdd100k" / "labels"

    train_src = img_root / "train"
    val_src = img_root / "val"
    train_json_src = labels_root / "bdd100k_labels_images_train.json"
    val_json_src = labels_root / "bdd100k_labels_images_val.json"

    missing = [str(p) for p in [train_src, val_src, train_json_src, val_json_src] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing:\n" + "\n".join(f"  {m}" for m in missing))

    train_dst = out / "images" / "100k" / "train"
    val_dst = out / "images" / "100k" / "val"
    labels_dst = out / "labels" / "det_20"
    labels_dst.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Flattening train images (trainA + trainB + loose) -> {train_dst}")
    subfolders = [d.name for d in train_src.iterdir() if d.is_dir()]
    print(f"      Subfolders found: {subfolders}")
    n_train = flatten_images(train_src, train_dst)
    print(f"      OK: {n_train} train images")

    print(f"\n[2/3] Copying val images -> {val_dst}")
    val_dst.mkdir(parents=True, exist_ok=True)
    n_val = 0
    for jpg in val_src.rglob("*.jpg"):
        dst = val_dst / jpg.name
        if not dst.exists():
            shutil.copy2(jpg, dst)
        n_val += 1
    print(f"      OK: {n_val} val images")

    print(f"\n[3/3] Copying labels")
    shutil.copy2(train_json_src, labels_dst / "det_train.json")
    print(f"      OK: det_train.json ({train_json_src.stat().st_size / 1e6:.1f} MB)")
    shutil.copy2(val_json_src, labels_dst / "det_val.json")
    print(f"      OK: det_val.json ({val_json_src.stat().st_size / 1e6:.1f} MB)")

    print("\n" + "="*60)
    print(f"Done! Upload this folder to Kaggle:")
    print(f"  {out}")
    print("\nStructure:")
    print("  images/100k/train/  [~63k .jpg files]")
    print("  images/100k/val/    [10k .jpg files]")
    print("  labels/det_20/det_train.json")
    print("  labels/det_20/det_val.json")
    print("\nIn notebook Cell 3, BDD100K_RAW = Path('/kaggle/input/<dataset-slug>')")


if __name__ == "__main__":
    main()
