"""Tests for YOLO wrapper — Section 18, CLAUDE.md.

Guards against the class-index mis-alignment bug where per-class AP was
assigned by positional order over TARGET_CLASSES instead of via Ultralytics'
results.box.ap_class_index. That bug silently mapped e.g. AP50_car → truck's
AP whenever a target class was absent from val — exactly the H1 scenario.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.model.yolo_wrapper import evaluate
from src.data.bdd100k import TARGET_CLASSES


def _mock_yolo_with_ap(ap_class_index, ap50, ap, p, r, map50=0.5, map_=0.4):
    """Build a mock YOLO instance whose .val() returns a results object
    with the given per-class arrays. All arrays must be same length.
    """
    box = SimpleNamespace(
        map50=map50,
        map=map_,
        ap_class_index=np.array(ap_class_index),
        ap50=np.array(ap50, dtype=float),
        ap=np.array(ap, dtype=float),
        p=np.array(p, dtype=float),
        r=np.array(r, dtype=float),
    )
    results = SimpleNamespace(box=box)
    model = MagicMock()
    model.val.return_value = results
    return model


def test_ap_mapped_by_class_index_not_position():
    """When val contains only classes {car, truck} (indices 0 and 2 out of
    [car, bus, truck, motorcycle]), Ultralytics returns
    ap_class_index=[0, 2] and per-class arrays of length 2. The wrapper must
    assign ap50[0] → car and ap50[1] → truck (NOT to bus).
    """
    model = _mock_yolo_with_ap(
        ap_class_index=[0, 2],
        ap50=[0.85, 0.60],
        ap=[0.70, 0.45],
        p=[0.80, 0.55],
        r=[0.75, 0.50],
    )
    metrics = evaluate(
        model=model, data_yaml=Path("/tmp/x.yaml"),
        img_size=640, conf=0.25, iou=0.7, device="cpu",
    )
    assert metrics["AP50_car"] == pytest.approx(0.85)
    assert metrics["AP50_truck"] == pytest.approx(0.60)
    # Missing classes must not be silently invented
    assert "AP50_bus" not in metrics
    assert "AP50_motorcycle" not in metrics
    # Precision + recall must also be per-class and correctly mapped
    assert metrics["precision_car"] == pytest.approx(0.80)
    assert metrics["recall_truck"] == pytest.approx(0.50)


def test_all_classes_present_maps_correctly():
    """When all 4 target classes are present, mapping order still comes from
    ap_class_index — Ultralytics may not sort by class index.
    """
    # Deliberately return in scrambled order to prove we don't rely on it
    model = _mock_yolo_with_ap(
        ap_class_index=[3, 0, 2, 1],  # motorcycle, car, truck, bus
        ap50=[0.30, 0.90, 0.55, 0.40],
        ap=[0.20, 0.75, 0.42, 0.30],
        p=[0.35, 0.85, 0.60, 0.45],
        r=[0.25, 0.80, 0.50, 0.35],
    )
    metrics = evaluate(
        model=model, data_yaml=Path("/tmp/x.yaml"),
        img_size=640, conf=0.25, iou=0.7, device="cpu",
    )
    assert metrics["AP50_motorcycle"] == pytest.approx(0.30)
    assert metrics["AP50_car"] == pytest.approx(0.90)
    assert metrics["AP50_truck"] == pytest.approx(0.55)
    assert metrics["AP50_bus"] == pytest.approx(0.40)


def test_empty_results_returns_empty_dict():
    model = MagicMock()
    model.val.return_value = None
    metrics = evaluate(
        model=model, data_yaml=Path("/tmp/x.yaml"),
        img_size=640, conf=0.25, iou=0.7, device="cpu",
    )
    assert metrics == {}


def test_out_of_range_class_index_is_ignored():
    """If Ultralytics ever returns an index >= len(TARGET_CLASSES),
    the wrapper must skip it instead of raising IndexError.
    """
    model = _mock_yolo_with_ap(
        ap_class_index=[0, 99],
        ap50=[0.80, 0.10],
        ap=[0.70, 0.05],
        p=[0.75, 0.15],
        r=[0.72, 0.12],
    )
    metrics = evaluate(
        model=model, data_yaml=Path("/tmp/x.yaml"),
        img_size=640, conf=0.25, iou=0.7, device="cpu",
    )
    assert metrics["AP50_car"] == pytest.approx(0.80)
    assert all(k.startswith(("mAP", "AP50_car", "AP_car", "precision_car", "recall_car"))
               for k in metrics.keys())


def test_target_classes_still_the_contract():
    """Regression guard: if TARGET_CLASSES order changes, mapping semantics
    change. Fail loudly so downstream analyses aren't silently invalidated.
    """
    assert TARGET_CLASSES == ("car", "bus", "truck", "motorcycle")


# ── Local-training val stub ──────────────────────────────────────────────────
# Ultralytics validates on the final epoch even with val=False and then runs
# final_eval — two passes over the GLOBAL val set per client per round. The
# returned weights come from the fp32 EMA snapshot, not best.pt, so the val
# set has no effect on them and can be a few training images.

def _write_dataset_yaml(root: Path, train_value: str) -> Path:
    import yaml
    data_yaml = root / "data_C0.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(root), "train": train_value, "val": "images/val",
        "nc": 4, "names": list(TARGET_CLASSES),
    }))
    return data_yaml


def test_val_stub_from_image_list(tmp_path):
    import yaml
    from src.model.yolo_wrapper import val_stub_data_yaml

    imgs = [str(tmp_path / "images" / "train" / f"{i}.jpg") for i in range(20)]
    (tmp_path / "C0_train.txt").write_text("\n".join(imgs) + "\n")
    src = _write_dataset_yaml(tmp_path, str(tmp_path / "C0_train.txt"))

    out = val_stub_data_yaml(src, tmp_path / "run", n_images=8)
    data = yaml.safe_load(out.read_text())
    stub = Path(data["val"]).read_text().split()
    assert stub == imgs[:8]
    assert {k: data[k] for k in ("path", "train", "nc", "names")} == \
        {k: yaml.safe_load(src.read_text())[k] for k in ("path", "train", "nc", "names")}
    assert yaml.safe_load(src.read_text())["val"] == "images/val"  # source untouched


def test_val_stub_from_image_directory(tmp_path):
    import yaml
    from src.model.yolo_wrapper import val_stub_data_yaml

    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    for name in ("b.jpg", "a.png", "c.jpg", "notes.txt"):
        (img_dir / name).write_bytes(b"")
    src = _write_dataset_yaml(tmp_path, "images/train")  # relative to `path`

    data = yaml.safe_load(val_stub_data_yaml(src, tmp_path / "run", n_images=2).read_text())
    stub = Path(data["val"]).read_text().split()
    assert stub == [str(img_dir / "a.png"), str(img_dir / "b.jpg")]


def test_val_stub_rejects_empty_train(tmp_path):
    from src.model.yolo_wrapper import val_stub_data_yaml

    (tmp_path / "empty.txt").write_text("")
    src = _write_dataset_yaml(tmp_path, str(tmp_path / "empty.txt"))
    with pytest.raises(ValueError, match="no training images"):
        val_stub_data_yaml(src, tmp_path / "run", n_images=8)
