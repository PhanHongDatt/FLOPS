from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# [ENGINEERING] Class order = YOLO index 0..3 (must match data.yaml names order)
# [LITERATURE] 4 vehicle classes per thesis Section 2
TARGET_CLASSES = ("car", "bus", "truck", "motorcycle")
CLASS_TO_IDX = {c: i for i, c in enumerate(TARGET_CLASSES)}

# [ENGINEERING] BDD100K annotation uses "motor" for motorcycles — map to canonical name
# Verified from actual annotation counts: car=713211, truck=29971, bus=11672, motor=3002
BDD_NAME_MAP: dict[str, str] = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "motor": "motorcycle",   # BDD100K raw name -> project canonical name
}


def load_bdd100k_annotations(ann_json: Path) -> list[dict[str, Any]]:
    with ann_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("frames", [])


def _normalize_category(raw_category: str) -> str | None:
    """Map raw BDD100K category name to project canonical name.

    Returns None if category is not in target set.
    """
    return BDD_NAME_MAP.get(raw_category)


def filter_target_classes(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only annotations containing at least one target class label.

    Also normalises category names via BDD_NAME_MAP (e.g. "motor"->"motorcycle").
    """
    result = []
    for frame in annotations:
        labels = frame.get("labels") or []
        kept = []
        for lb in labels:
            canonical = _normalize_category(lb.get("category", ""))
            if canonical is not None:
                kept.append({**lb, "category": canonical})
        if kept:
            result.append({**frame, "labels": kept})
    return result


def annotation_to_yolo_line(
    label: dict[str, Any],
    img_w: int,
    img_h: int,
) -> str | None:
    cat = label.get("category")
    if cat not in CLASS_TO_IDX:
        return None
    box2d = label.get("box2d")
    if box2d is None:
        return None
    x1, y1, x2, y2 = box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    cls_idx = CLASS_TO_IDX[cat]
    return f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def write_yolo_labels(
    annotations: list[dict[str, Any]],
    label_dir: Path,
    img_w: int = 1280,
    img_h: int = 720,
) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    for frame in annotations:
        name = Path(frame["name"]).stem
        lines = [
            annotation_to_yolo_line(lb, img_w, img_h)
            for lb in frame.get("labels", [])
        ]
        lines = [ln for ln in lines if ln is not None]
        (label_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")


class BDD100KDetectionDataset:
    """Minimal dataset wrapper returning (image_path, label_path) pairs.

    Does NOT inherit torch.utils.data.Dataset to avoid requiring torch
    at import time. Duck-type compatible for iteration.
    """

    def __init__(self, image_dir: Path, label_dir: Path) -> None:
        self.samples = sorted(
            [
                (img, label_dir / (img.stem + ".txt"))
                for img in image_dir.glob("*.jpg")
                if (label_dir / (img.stem + ".txt")).exists()
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Path, Path]:
        return self.samples[idx]
