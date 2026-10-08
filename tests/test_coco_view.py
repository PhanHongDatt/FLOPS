"""ADR-018 — COCO-id view of the YOLO tree for models that keep the 80-class COCO head."""
from __future__ import annotations

from pathlib import Path

import yaml

from src.data.coco_view import COCO_IDS, coco_data_yaml, ensure_coco_root, remap_label_text


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "bdd100k_yolo"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / f"{split}1.jpg").write_bytes(b"x")
        (root / "labels" / split / f"{split}1.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n2 0.3 0.3 0.1 0.1\n3 0.4 0.4 0.1 0.1")
    return root


def test_remap_uses_coco_ids():
    out = remap_label_text("0 0.5 0.5 0.1 0.1\n1 1 1 1 1\n2 2 2 2 2\n3 3 3 3 3\n")
    assert [int(ln.split()[0]) for ln in out.splitlines()] == [COCO_IDS["car"], COCO_IDS["bus"],
                                                                COCO_IDS["truck"], COCO_IDS["motorcycle"]]
    assert COCO_IDS == {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}


def test_view_has_own_labels_and_same_images(tmp_path):
    root = _tree(tmp_path)
    coco = ensure_coco_root(root)
    assert coco.name == "bdd100k_yolo_coco"
    lbl = (coco / "labels" / "train" / "train1.txt").read_text()
    assert [int(ln.split()[0]) for ln in lbl.splitlines()] == [2, 5, 7, 3]
    assert (root / "labels" / "train" / "train1.txt").read_text().startswith("0 ")   # original untouched
    assert (coco / "images" / "val" / "val1.jpg").exists()
    assert ensure_coco_root(root) == coco                                          # idempotent


def test_data_yaml_rewrite_points_lists_at_the_view(tmp_path):
    root = _tree(tmp_path)
    lst = tmp_path / "C0_train.txt"
    lst.write_text(str((root / "images" / "train" / "train1.jpg").resolve()) + "\n")
    dy = tmp_path / "data_C0.yaml"
    dy.write_text(yaml.safe_dump({"path": str(root), "train": str(lst), "val": "images/val", "nc": 4,
                                  "names": ["car", "bus", "truck", "motorcycle"]}))
    names = [f"c{i}" for i in range(80)]
    out = yaml.safe_load(coco_data_yaml(dy, tmp_path / "view", names=names).read_text())
    assert out["nc"] == 80 and out["names"] == names
    assert Path(out["path"]).name == "bdd100k_yolo_coco"
    # s10_g P3 v1: a directory val entry was .resolve()d by Ultralytics back to the original labels
    for split, name in (("train", "train1.jpg"), ("val", "val1.jpg")):
        lst = Path(out[split])
        assert lst.is_file(), f"{split} must be an image list, never a directory"
        line = lst.read_text().strip()
        assert "bdd100k_yolo_coco" in line and line.endswith(name)
        label = Path(line.replace(f"{__import__('os').sep}images{__import__('os').sep}",
                                  f"{__import__('os').sep}labels{__import__('os').sep}")).with_suffix(".txt")
        assert int(label.read_text().split()[0]) == 2          # car remapped to the COCO id
