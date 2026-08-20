"""Tests for class statistics — Section 18, CLAUDE.md."""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.data.class_stats import compute_class_stats
from src.data.bdd100k import TARGET_CLASSES


def _write_labels(label_dir: Path, filename: str, class_ids: list[int]) -> None:
    lines = [f"{c} 0.5 0.5 0.2 0.2" for c in class_ids]
    (label_dir / filename).write_text("\n".join(lines))


def test_zero_boxes_means_missing():
    with tempfile.TemporaryDirectory() as tmp:
        label_dir = Path(tmp)
        _write_labels(label_dir, "a.txt", [0, 0])   # car x2
        _write_labels(label_dir, "b.txt", [1])       # bus x1
        # truck=2, motorcycle=3 have zero boxes

        stats = compute_class_stats(
            label_dir, client_id="C0", round_id=1,
            rare_rule_id="test_v1", rare_threshold=5,
        )

    assert "truck" in stats.missing_classes
    assert "motorcycle" in stats.missing_classes
    assert stats.boxes_per_class["car"] == 2
    assert stats.boxes_per_class["bus"] == 1


def test_counts_correct():
    with tempfile.TemporaryDirectory() as tmp:
        label_dir = Path(tmp)
        _write_labels(label_dir, "x.txt", [0, 1, 2, 3])
        _write_labels(label_dir, "y.txt", [0, 0])

        stats = compute_class_stats(
            label_dir, client_id="C1", round_id=2,
            rare_rule_id="test_v1", rare_threshold=2,
        )

    assert stats.boxes_per_class["car"] == 3
    assert stats.boxes_per_class["bus"] == 1
    assert stats.total_boxes == 6
    assert stats.num_images == 2


def test_present_and_missing_disjoint():
    with tempfile.TemporaryDirectory() as tmp:
        label_dir = Path(tmp)
        _write_labels(label_dir, "z.txt", [0])

        stats = compute_class_stats(
            label_dir, client_id="C2", round_id=0,
            rare_rule_id="test_v1", rare_threshold=10,
        )

    assert set(stats.present_classes).isdisjoint(set(stats.missing_classes))
    assert set(stats.present_classes) | set(stats.missing_classes) == set(TARGET_CLASSES)


def test_rare_class_pre_registered():
    """Rare threshold must be set before inspecting outcomes."""
    with tempfile.TemporaryDirectory() as tmp:
        label_dir = Path(tmp)
        _write_labels(label_dir, "r.txt", [2, 2])   # truck x2

        stats = compute_class_stats(
            label_dir, client_id="C3", round_id=1,
            rare_rule_id="pre_registered_rule_v1", rare_threshold=3,
        )

    assert "truck" in stats.rare_classes
    assert stats.rare_rule_id == "pre_registered_rule_v1"
