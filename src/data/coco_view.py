"""COCO-id view of a prepared YOLO dataset, for models that keep the 80-class COCO head (ADR-018).

The project's YOLO tree labels the targets 0..3 (car, bus, truck, motorcycle). A model that keeps the
pretrained COCO head needs the COCO ids instead (car 2, motorcycle 3, bus 5, truck 7). This builds a
sibling tree ``<yolo_root>_coco`` with the SAME images (directory links) and remapped label files in
its own folders — Ultralytics keys its label cache on the label folder, so the two views never share a
cache — and rewrites data.yaml files (and their image lists) to point at it.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from src.data.bdd100k import TARGET_CLASSES

COCO_IDS: dict[str, int] = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
REMAP: dict[int, int] = {i: COCO_IDS[c] for i, c in enumerate(TARGET_CLASSES)}
_MARKER = ".coco_view_done"


def coco_names() -> list[str]:
    """The 80 COCO class names in index order (from the Ultralytics dataset config)."""
    from ultralytics.utils import ROOT
    cfg = yaml.safe_load((ROOT / "cfg" / "datasets" / "coco.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    return [names[i] for i in range(len(names))]


def remap_label_text(text: str) -> str:
    out = []
    for ln in text.splitlines():
        parts = ln.split()
        if parts:
            out.append(" ".join([str(REMAP[int(parts[0])])] + parts[1:]))
    return "\n".join(out)


def _link_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src.resolve(), dst, target_is_directory=True)
    except OSError:                       # e.g. Windows without symlink rights
        shutil.copytree(src, dst)


def ensure_coco_root(yolo_root: Path) -> Path:
    """Build (once) ``<yolo_root>_coco`` with linked images and COCO-id labels for train and val."""
    yolo_root = Path(yolo_root).resolve()
    root = yolo_root.parent / f"{yolo_root.name}_coco"
    if (root / _MARKER).exists():
        return root
    for split in ("train", "val"):
        src_img = yolo_root / "images" / split
        if not src_img.is_dir():
            continue
        (root / "images").mkdir(parents=True, exist_ok=True)
        _link_dir(src_img, root / "images" / split)
        dst_lbl = root / "labels" / split
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for f in (yolo_root / "labels" / split).glob("*.txt"):
            (dst_lbl / f.name).write_text(remap_label_text(f.read_text(encoding="utf-8")), encoding="utf-8")
    (root / _MARKER).write_text("ok", encoding="utf-8")
    return root


def coco_data_yaml(data_yaml: Path, out_dir: Path, names: list[str] | None = None) -> Path:
    """Copy of ``data_yaml`` that reads the same images through the COCO-id view.

    ``train``/``val`` entries become image-list files whose paths go through ``<root>_coco/images``:
    list files are rewritten line by line, directory entries are expanded into a list. A directory is
    never passed on, because Ultralytics resolves it (and with it the link back to the original labels).
    """
    data_yaml = Path(data_yaml)
    data: dict[str, Any] = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path") or data_yaml.parent).resolve()
    coco_root = ensure_coco_root(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    new = {**data, "path": str(coco_root), "nc": 80, "names": names or coco_names()}
    for split in ("train", "val"):
        entry = data.get(split)
        if entry is None:
            continue
        p = Path(entry)
        p = p if p.is_absolute() else root / p
        old_prefix, new_prefix = str(root / "images"), str(coco_root / "images")
        if p.is_dir():
            # Never hand Ultralytics a directory here: check_det_dataset() calls .resolve() on it,
            # which follows the images/<split> link back to the ORIGINAL tree and so reads the
            # original 0..3 labels (s10_g P3 v1: every val metric except motorcycle was wrong).
            # An image list is not resolved line by line, so labels come from the COCO view.
            rel = p.relative_to(root / "images")
            lines = [str(coco_root / "images" / rel / f.name) for f in sorted(p.iterdir())
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
            lst = out_dir / f"{data_yaml.stem}_{split}_coco.txt"
            lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
            new[split] = str(lst)
        elif p.is_file():
            lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            lst = out_dir / f"{data_yaml.stem}_{split}_coco.txt"
            lst.write_text("\n".join(ln.replace(old_prefix, new_prefix, 1) for ln in lines) + "\n", encoding="utf-8")
            new[split] = str(lst)
    out = out_dir / f"{data_yaml.stem}_coco.yaml"
    out.write_text(yaml.safe_dump(new), encoding="utf-8")
    return out
