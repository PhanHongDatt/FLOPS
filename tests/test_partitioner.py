"""Tests for data partitioning — Section 18, CLAUDE.md."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.data.partitioner import (
    partition_iid,
    partition_missing_class,
    save_partition_manifest,
    load_partition_manifest,
)
from src.data.bdd100k import TARGET_CLASSES, BDD_NAME_MAP, filter_target_classes


def _make_label_dir(tmp: Path, images: list[str], class_map: dict[str, list[int]]) -> Path:
    label_dir = tmp / "labels"
    label_dir.mkdir()
    for img in images:
        stem = Path(img).stem
        classes = class_map.get(img, [])
        lines = [f"{c} 0.5 0.5 0.2 0.2" for c in classes]
        (label_dir / f"{stem}.txt").write_text("\n".join(lines))
    return label_dir


def test_iid_partition_all_images_assigned():
    images = [f"img_{i:04d}.jpg" for i in range(20)]
    with tempfile.TemporaryDirectory() as tmp:
        label_dir = _make_label_dir(Path(tmp), images, {img: [0] for img in images})
        manifest = partition_iid(images, label_dir, num_clients=4, seed=42, partition_id="s0_test")

    all_assigned = sum(len(v) for v in manifest.client_assignments.values())
    assert all_assigned == len(images)


def test_iid_partition_deterministic():
    images = [f"img_{i:04d}.jpg" for i in range(20)]
    with tempfile.TemporaryDirectory() as tmp:
        label_dir = _make_label_dir(Path(tmp), images, {img: [0] for img in images})
        m1 = partition_iid(images, label_dir, num_clients=4, seed=42, partition_id="p1")
        m2 = partition_iid(images, label_dir, num_clients=4, seed=42, partition_id="p2")

    assert m1.client_assignments == m2.client_assignments


def test_iid_partition_different_seeds():
    images = [f"img_{i:04d}.jpg" for i in range(40)]
    with tempfile.TemporaryDirectory() as tmp:
        label_dir = _make_label_dir(Path(tmp), images, {img: [0] for img in images})
        m1 = partition_iid(images, label_dir, num_clients=4, seed=42, partition_id="p1")
        m2 = partition_iid(images, label_dir, num_clients=4, seed=99, partition_id="p2")

    assert m1.client_assignments != m2.client_assignments


def test_missing_class_target_truly_absent():
    """Section 18: target missing class truly has zero boxes."""
    images = [f"img_{i:04d}.jpg" for i in range(40)]
    class_map = {}
    for i, img in enumerate(images):
        # half have class 0 (car), half have class 1 (bus)
        class_map[img] = [0] if i < 20 else [1]

    with tempfile.TemporaryDirectory() as tmp:
        label_dir = _make_label_dir(Path(tmp), images, class_map)
        manifest = partition_missing_class(
            images,
            label_dir,
            num_clients=4,
            missing_map={"C0": ["car"], "C1": ["bus"]},
            seed=42,
            partition_id="s1_test",
        )

    assert "car" in manifest.missing_classes.get("C0", [])
    assert "bus" in manifest.missing_classes.get("C1", [])


def test_manifest_save_load_roundtrip():
    images = [f"img_{i:04d}.jpg" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        label_dir = _make_label_dir(tmp_path, images, {img: [0] for img in images})
        manifest = partition_iid(images, label_dir, num_clients=2, seed=0, partition_id="roundtrip")

        out = tmp_path / "manifest.yaml"
        save_partition_manifest(manifest, out)
        loaded = load_partition_manifest(out)

    assert loaded.partition_id == manifest.partition_id
    assert loaded.seed == manifest.seed
    assert loaded.client_assignments == manifest.client_assignments


def test_motor_mapped_to_motorcycle():
    """BDD100K annotation uses 'motor'; must be normalised to 'motorcycle'."""
    assert BDD_NAME_MAP["motor"] == "motorcycle"

    frames = [
        {
            "name": "img_0000.jpg",
            "labels": [
                {"category": "motor", "box2d": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
                {"category": "car",   "box2d": {"x1": 20, "y1": 0, "x2": 50, "y2": 30}},
                {"category": "person","box2d": {"x1": 60, "y1": 0, "x2": 80, "y2": 40}},
            ],
        }
    ]
    filtered = filter_target_classes(frames)
    assert len(filtered) == 1
    cats = {lb["category"] for lb in filtered[0]["labels"]}
    assert "motorcycle" in cats, "motor should be mapped to motorcycle"
    assert "motor" not in cats, "raw 'motor' should not appear after normalisation"
    assert "person" not in cats, "person is not a target class"
    assert "car" in cats


def test_missing_class_drops_ineligible_images():
    """CLAUDE.md §22: an image containing ONLY classes excluded by every
    client must NOT be silently assigned — the previous fallback would break
    the S1 missing-class contract. It should be dropped instead.
    """
    images = [f"img_{i:04d}.jpg" for i in range(10)]
    # Every image has only class 0 (car). If both clients exclude 'car',
    # no image is eligible.
    class_map = {img: [0] for img in images}
    with tempfile.TemporaryDirectory() as tmp:
        label_dir = _make_label_dir(Path(tmp), images, class_map)
        manifest = partition_missing_class(
            images,
            label_dir,
            num_clients=2,
            missing_map={"C0": ["car"], "C1": ["car"]},
            seed=42,
            partition_id="s1_drop_test",
        )
    # All 10 images should be dropped, both clients truly car-free
    assigned_total = sum(len(v) for v in manifest.client_assignments.values())
    assert assigned_total == 0, f"Expected 0 assignments, got {assigned_total}"
    assert manifest.class_counts["C0"]["car"] == 0
    assert manifest.class_counts["C1"]["car"] == 0


def test_missing_class_partial_overlap_keeps_eligible_only():
    """When some clients still allow the class, images should go there."""
    images = [f"img_{i:04d}.jpg" for i in range(20)]
    class_map = {img: [0] for img in images}  # all car
    with tempfile.TemporaryDirectory() as tmp:
        label_dir = _make_label_dir(Path(tmp), images, class_map)
        manifest = partition_missing_class(
            images,
            label_dir,
            num_clients=3,
            missing_map={"C0": ["car"], "C1": ["car"], "C2": []},
            seed=42,
            partition_id="s1_partial_test",
        )
    # All 20 must land on C2 (only eligible client); C0 and C1 stay empty
    assert len(manifest.client_assignments["C0"]) == 0
    assert len(manifest.client_assignments["C1"]) == 0
    assert len(manifest.client_assignments["C2"]) == 20
    assert manifest.class_counts["C0"]["car"] == 0
    assert manifest.class_counts["C1"]["car"] == 0
