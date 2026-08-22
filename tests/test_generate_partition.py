"""Tests for scripts.generate_partition — Section 18, CLAUDE.md."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Ensure repo root on sys.path so `scripts` package is importable in test env
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.generate_partition import generate_partition_artifacts  # noqa: E402
from src.data.bdd100k import TARGET_CLASSES  # noqa: E402
from src.data.partitioner import load_partition_manifest  # noqa: E402


def _make_yolo_root(tmp: Path, images: list[str], class_map: dict[str, list[int]]) -> Path:
    """Create a minimal YOLO-formatted root: images/train + labels/train."""
    root = tmp / "yolo_root"
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    (root / "images" / "val").mkdir(parents=True)  # required val dir (empty ok for test)
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    for img in images:
        # empty jpg placeholder — content not read, only listed
        (img_dir / img).write_bytes(b"")
        stem = Path(img).stem
        classes = class_map.get(img, [])
        lines = [f"{c} 0.5 0.5 0.2 0.2" for c in classes]
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
    return root


def _write_config(tmp: Path, config: dict) -> Path:
    p = tmp / "partition_config.yaml"
    with p.open("w") as f:
        yaml.dump(config, f)
    return p


def test_s0_iid_produces_manifest_and_data_yamls():
    images = [f"img_{i:04d}.jpg" for i in range(20)]
    class_map = {img: [0] for img in images}  # all car
    config = {
        "partition_id": "s0_test",
        "scenario": "S0",
        "seed": 42,
        "num_clients": 4,
        "missing_map": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        yolo_root = _make_yolo_root(tmp_path, images, class_map)
        cfg_path = _write_config(tmp_path, config)
        out_dir = tmp_path / "partitions"

        manifest_path, data_yamls = generate_partition_artifacts(
            partition_config_path=cfg_path,
            yolo_root=yolo_root,
            output_dir=out_dir,
        )

        assert manifest_path.exists()
        assert manifest_path.name == "manifest.yaml"
        assert set(data_yamls.keys()) == {"C0", "C1", "C2", "C3"}
        for p in data_yamls.values():
            assert p.exists()

        # Manifest loadable via existing loader
        manifest = load_partition_manifest(manifest_path)
        assert manifest.partition_id == "s0_test"
        assert manifest.scenario == "S0"
        total = sum(len(v) for v in manifest.client_assignments.values())
        assert total == len(images)


def test_s1_missing_class_target_truly_absent():
    images = [f"img_{i:04d}.jpg" for i in range(40)]
    class_map = {img: [0] if i < 20 else [1] for i, img in enumerate(images)}
    config = {
        "partition_id": "s1_test",
        "scenario": "S1",
        "seed": 42,
        "num_clients": 4,
        "missing_map": {"C0": ["car"], "C1": ["bus"]},
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        yolo_root = _make_yolo_root(tmp_path, images, class_map)
        cfg_path = _write_config(tmp_path, config)

        manifest_path, _ = generate_partition_artifacts(
            partition_config_path=cfg_path,
            yolo_root=yolo_root,
            output_dir=tmp_path / "partitions",
        )
        manifest = load_partition_manifest(manifest_path)

        assert "car" in manifest.missing_classes.get("C0", [])
        assert "bus" in manifest.missing_classes.get("C1", [])
        assert manifest.class_counts["C0"]["car"] == 0
        assert manifest.class_counts["C1"]["bus"] == 0


def test_client_data_yaml_schema():
    images = [f"img_{i:04d}.jpg" for i in range(8)]
    class_map = {img: [0] for img in images}
    config = {
        "partition_id": "schema_test",
        "scenario": "S0",
        "seed": 0,
        "num_clients": 2,
        "missing_map": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        yolo_root = _make_yolo_root(tmp_path, images, class_map)
        cfg_path = _write_config(tmp_path, config)

        _, data_yamls = generate_partition_artifacts(
            partition_config_path=cfg_path,
            yolo_root=yolo_root,
            output_dir=tmp_path / "partitions",
        )

        with data_yamls["C0"].open() as f:
            data = yaml.safe_load(f)

        # YOLO data.yaml required fields
        assert set(["path", "train", "val", "nc", "names"]).issubset(data.keys())
        assert data["nc"] == len(TARGET_CLASSES)
        assert data["names"] == list(TARGET_CLASSES)
        # train points to a real image-list file
        train_list = Path(data["train"])
        assert train_list.exists()
        listed = train_list.read_text().strip().splitlines()
        assert len(listed) > 0
        # each listed line is an absolute path to a .jpg
        for line in listed:
            assert Path(line).suffix == ".jpg"


def test_unknown_scenario_raises():
    config = {
        "partition_id": "bad",
        "scenario": "S99",
        "seed": 0,
        "num_clients": 2,
        "missing_map": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        yolo_root = _make_yolo_root(tmp_path, ["a.jpg"], {"a.jpg": [0]})
        cfg_path = _write_config(tmp_path, config)
        with pytest.raises(ValueError, match="Unknown scenario"):
            generate_partition_artifacts(
                partition_config_path=cfg_path,
                yolo_root=yolo_root,
                output_dir=tmp_path / "partitions",
            )


def test_missing_yolo_root_raises():
    config = {
        "partition_id": "no_data",
        "scenario": "S0",
        "seed": 0,
        "num_clients": 2,
        "missing_map": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg_path = _write_config(tmp_path, config)
        with pytest.raises(FileNotFoundError):
            generate_partition_artifacts(
                partition_config_path=cfg_path,
                yolo_root=tmp_path / "nonexistent",
                output_dir=tmp_path / "partitions",
            )
