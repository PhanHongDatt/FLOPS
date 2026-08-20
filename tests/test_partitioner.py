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
from src.data.bdd100k import TARGET_CLASSES


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
