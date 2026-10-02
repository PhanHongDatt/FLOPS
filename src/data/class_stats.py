from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.data.bdd100k import TARGET_CLASSES


@dataclass(frozen=True)
class ClientClassStats:
    client_id: str
    round_id: int
    num_images: int
    total_boxes: int
    boxes_per_class: dict[str, int]
    present_classes: tuple[str, ...]
    missing_classes: tuple[str, ...]
    rare_classes: tuple[str, ...]
    rare_rule_id: str
    missing_threshold: int = 0  # zero positive boxes = missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "round_id": self.round_id,
            "num_images": self.num_images,
            "total_boxes": self.total_boxes,
            "boxes_per_class": dict(self.boxes_per_class),
            "present_classes": list(self.present_classes),
            "missing_classes": list(self.missing_classes),
            "rare_classes": list(self.rare_classes),
            "definition": {
                "missing_threshold": self.missing_threshold,
                "rare_rule_id": self.rare_rule_id,
            },
        }


def compute_class_stats(
    label_dir: Path,
    client_id: str,
    round_id: int,
    rare_rule_id: str,
    rare_threshold: int,
    missing_threshold: int = 0,
) -> ClientClassStats:
    """Compute per-class box counts from YOLO label files.

    rare_threshold and rare_rule_id must be pre-registered before
    model outcomes are inspected (Section 9, CLAUDE.md).
    """
    boxes: dict[str, int] = {c: 0 for c in TARGET_CLASSES}
    num_images = 0

    for label_file in label_dir.glob("*.txt"):
        lines = label_file.read_text(encoding="utf-8").strip().splitlines()
        valid = [l for l in lines if l.strip()]
        if not valid:
            continue
        num_images += 1
        for line in valid:
            parts = line.split()
            if not parts:
                continue
            cls_idx = int(parts[0])
            if cls_idx < len(TARGET_CLASSES):
                boxes[TARGET_CLASSES[cls_idx]] += 1

    present = tuple(c for c in TARGET_CLASSES if boxes[c] > missing_threshold)
    missing = tuple(c for c in TARGET_CLASSES if boxes[c] <= missing_threshold)
    rare = tuple(c for c in TARGET_CLASSES if 0 < boxes[c] <= rare_threshold)

    return ClientClassStats(
        client_id=client_id,
        round_id=round_id,
        num_images=num_images,
        total_boxes=sum(boxes.values()),
        boxes_per_class=boxes,
        present_classes=present,
        missing_classes=missing,
        rare_classes=rare,
        rare_rule_id=rare_rule_id,
        missing_threshold=missing_threshold,
    )


def save_class_stats(stats: ClientClassStats, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(stats.to_dict(), f, default_flow_style=False, allow_unicode=True)
